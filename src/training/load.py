"""
Data loading & splitting utilities for MovieLens-10M.

Key function:
    prepare_datasets(...)
        -> returns (train, val, test) as Spotlight Interactions objects.
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import sqlite3
import time
import re
import warnings

# Fix OpenMP / MKL issues on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch

torch.set_num_threads(1)

import numpy as np
import pandas as pd

from spotlight.interactions import Interactions
from spotlight.cross_validation import user_based_train_test_split
from spotlight.evaluation import mrr_score, precision_recall_score
from spotlight.factorization.implicit import ImplicitFactorizationModel

# Project root: .../team-A2-recsys/
ROOT = Path(__file__).resolve().parents[2]

#RATINGS_PATH = ROOT / "data" / "raw" / "ml-10M100K" / "ratings.dat"
RATINGS_PATH = ROOT / "data" / "processed" / "ml-10M100K" / "ratings_clean.dat"
MOVIES_PATH  = ROOT / "data" / "processed" / "ml-10M100K" / "movies.dat"
DEFAULT_DB_PATH = ROOT / "src" / "app" / "user_data.db"

def get_frontend_ratings(db_path: str | Path) -> pd.DataFrame:
    """
    Extract ratings only from frontend users who finished initial ratings.
    Returns: username, movie (title without year), rating
    """
    conn = sqlite3.connect(str(db_path))
    query = """
        SELECT r.username, r.movie, r.rating
        FROM ratings r
        JOIN initial_ratings i ON r.username = i.username
        WHERE i.done = 1
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def _normalize_title(s: str) -> str:
    """
    Normalize titles so DB titles (no year) can match movies.dat titles (with year).
    Removes trailing ' (YYYY)' from MovieLens titles.
    """
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s*\(\d{4}\)\s*$", "", s)   # remove trailing (YYYY)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def load_movies(movies_path: str | Path) -> pd.DataFrame:
    """
    Load movies.dat (CSV): movie_id,title,genres
    """
    df = pd.read_csv(movies_path)
    if not {"movie_id", "title"}.issubset(df.columns):
        raise ValueError(
            f"{movies_path} must contain columns at least movie_id and title. Found: {list(df.columns)}"
        )
    df["title_norm"] = df["title"].map(_normalize_title)
    return df


def convert_frontend_to_movielens_schema(
    db_path: str | Path,
    movies_path: str | Path,
    base_max_user_id: int,
) -> pd.DataFrame:
    """
    Convert frontend DB ratings into MovieLens-like schema:
    user_id (numeric), item_id (=movie_id numeric), rating, timestamp.

    - username -> numeric user_id appended after MovieLens max user_id
    - movie title (no year) -> movie_id using movies.dat
    - timestamp -> placeholder 'now' for all rows (DB has no timestamp)
    """
    db_ratings = get_frontend_ratings(db_path)
    if db_ratings.empty:
        return pd.DataFrame(columns=["user_id", "item_id", "rating", "timestamp"])

    movies_df = load_movies(movies_path)

    # Normalize titles for join
    db_ratings = db_ratings.copy()
    db_ratings["movie_norm"] = db_ratings["movie"].map(_normalize_title)

    # Drop ambiguous normalized titles in MovieLens (same title across years/IDs)
    counts = movies_df.groupby("title_norm")["movie_id"].nunique()
    ambiguous = set(counts[counts > 1].index)
    amb_in_db = sorted(set(db_ratings["movie_norm"]) & ambiguous)
    if amb_in_db:
        warnings.warn(
            f"Ambiguous titles detected for {len(amb_in_db)} normalized titles. "
            f"Dropping those DB rows. Examples: {amb_in_db[:5]}"
        )
        db_ratings = db_ratings[~db_ratings["movie_norm"].isin(ambiguous)].copy()

    merged = db_ratings.merge(
        movies_df[["movie_id", "title_norm"]],
        left_on="movie_norm",
        right_on="title_norm",
        how="left",
    )

    # Drop unmatched titles (should be rare based on your confirmation)
    missing = merged["movie_id"].isna().sum()
    if missing:
        examples = merged.loc[merged["movie_id"].isna(), "movie"].dropna().unique()[:5]
        warnings.warn(
            f"{missing} DB ratings could not be mapped to movie_id. Dropping them. Examples: {examples}"
        )
        merged = merged.dropna(subset=["movie_id"]).copy()

    # username -> new numeric user_id (deterministic ordering)
    usernames = sorted(merged["username"].astype(str).unique())
    user_map = {u: base_max_user_id + 1 + i for i, u in enumerate(usernames)}
    merged["user_id"] = merged["username"].map(user_map).astype("int64")

    # Placeholder timestamp (since DB has none)
    ts = int(time.time())
    merged["timestamp"] = ts

    # Align with ratings.dat schema (item_id == movie_id)
    df = pd.DataFrame({
        "user_id": merged["user_id"].astype("int64"),
        "item_id": merged["movie_id"].astype("int64"),
        "rating":  merged["rating"].astype("float32"),
        "timestamp": merged["timestamp"].astype("int64"),
    })

    # Keep newest rating if duplicates exist (best approach you requested)
    df = df.sort_values("timestamp").drop_duplicates(["user_id", "item_id"], keep="last")
    return df


def load_ratings(path: str | Path) -> pd.DataFrame:
    """Load the MovieLens ratings.dat file into a pandas DataFrame."""
    cols = ["user_id", "movie_id", "rating", "timestamp"]

    df = pd.read_csv(
        path,
        #sep="::",
        sep=",",
        header=0,
        engine="python",
        #names=cols,
        dtype={
            "user_id": "Int64",
            "movie_id": "Int64",
            "rating": "float32",
            "timestamp": "Int64",
        },
        on_bad_lines="skip",
    )

    # Drop any rows with missing values
    df = df.dropna(subset=["user_id", "movie_id", "rating", "timestamp"])

    return df


def build_interactions(df: pd.DataFrame, return_mappings: bool = False):
    """
    Convert a ratings DataFrame into a Spotlight Interactions object.

    If return_mappings=True, also returns the arrays of original user and item IDs.
    """
    # Factorize gives integer-coded ids and mapping arrays
    user_ids, user_unique = pd.factorize(df["user_id"])
    item_ids, item_unique = pd.factorize(df["movie_id"])

    user_ids = user_ids.astype("int32")
    item_ids = item_ids.astype("int32")
    ratings = df["rating"].astype("float32").values
    timestamps = df["timestamp"].astype("int64").astype("int32")

    interactions = Interactions(
        user_ids=user_ids,
        item_ids=item_ids,
        ratings=ratings,
        timestamps=timestamps,
        num_users=len(user_unique),
        num_items=len(item_unique),
    )

    if return_mappings:
        # user_unique and item_unique are arrays of original IDs
        return interactions, user_unique, item_unique

    return interactions


def prepare_datasets(
    ratings_path: str | Path = RATINGS_PATH,
    movies_path: str | Path = MOVIES_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
    include_frontend: bool = False,
    sample_size: Optional[int] = None,
    user_test_percentage: float = 0.10,
    user_val_percentage_from_train: float = 0.111,
    seed: int = 42,
) -> Tuple[Interactions, Interactions, Interactions]:
    """
    Load MovieLens ratings, optionally subsample, build Interactions,
    and return (train, val, test) user-based splits.

    Parameters
    ----------
    ratings_path:
        Path to ratings.dat.
    sample_size:
        If not None, randomly sample this many rows from the full dataset.
        Using the same `seed` ensures reproducible sampling.
    user_test_percentage:
        Fraction of each user's interactions to hold out for test.
    user_val_percentage_from_train:
        Fraction of the remaining train_val interactions to hold out for val.
    seed:
        Seed for all random operations (sampling + splits). Using the same
        seed means data splits are identical across all model runs.

    Returns
    -------
    train, val, test : spotlight.interactions.Interactions
    """
    ratings_path = Path(ratings_path)

    print(f"Loading ratings from {ratings_path} ...")
    df = load_ratings(ratings_path)

    if include_frontend:
        max_user = int(df["user_id"].max())
        frontend_df = convert_frontend_to_movielens_schema(
            db_path=db_path,
            movies_path=movies_path,
            base_max_user_id=max_user,
        )
        if not frontend_df.empty:
            df = pd.concat([df, frontend_df], ignore_index=True)
            # In case duplicates exist across merged data, keep newest (by timestamp)
            df = df.sort_values("timestamp").drop_duplicates(["user_id", "item_id"], keep="last")

    # Optional subsampling for fast debugging
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=seed)
        print(f"Loaded {len(df):,} interactions (sampled)")
    else:
        print(f"Loaded {len(df):,} interactions (FULL dataset)")

    print("Building Spotlight Interactions object ...")
    interactions = build_interactions(df)

    print("Splitting into train, validation and test sets (user-based) ...")

    # Two RNGs so that val and test splits are independent but reproducible
    rng_test = np.random.RandomState(seed)
    rng_val = np.random.RandomState(seed + 1)

    train_val, test = user_based_train_test_split(
        interactions,
        test_percentage=user_test_percentage,
        random_state=rng_test,
    )

    train, val = user_based_train_test_split(
        train_val,
        test_percentage=user_val_percentage_from_train,
        random_state=rng_val,
    )

    print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")

    return train, val, test

def load_full_dataframe(
    ratings_path: str | Path = RATINGS_PATH,
    movies_path: str | Path = MOVIES_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
    include_frontend: bool = False,
) -> pd.DataFrame:
    """
    Return the FULL merged ratings dataframe (before train/val/test split).
    """
    df = load_ratings(ratings_path)

    if include_frontend:
        max_user = int(df["user_id"].max())
        frontend_df = convert_frontend_to_movielens_schema(
            db_path=db_path,
            movies_path=movies_path,
            base_max_user_id=max_user,
        )
        if not frontend_df.empty:
            df = pd.concat([df, frontend_df], ignore_index=True)
            df = df.sort_values("timestamp").drop_duplicates(
                ["user_id", "item_id"], keep="last"
            )
        print("Number of unique users in full dataframe:")
        print(df["user_id"].nunique())

    return df


def main() -> None:
    """
    Quick sanity check: load data and train a tiny implicit BPR model.

    This is just for local testing and is not used by the main training pipeline.
    """
    train, val, test = prepare_datasets()

    print(
        "Training small implicit BPR factorization model to test data is "
        "formatted correctly..."
    )
    model = ImplicitFactorizationModel(
        loss="bpr",
        embedding_dim=16,
        n_iter=5,
        batch_size=2048,
        learning_rate=5e-3,
        l2=1e-6,
        num_negative_samples=3,
        use_cuda=False,
        random_state=np.random.RandomState(42),
    )

    model.fit(train, verbose=True)

    print("Evaluating model ...")
    mrr = mrr_score(model, test, train=train).mean()
    precision, recall = precision_recall_score(model, test, train=train, k=10)
    print(f"MRR:            {mrr:.4f}")
    print(f"Precision@10:   {precision.mean():.4f}")
    print(f"Recall@10:      {recall.mean():.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
