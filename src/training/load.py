"""
Data loading & splitting utilities for MovieLens-10M.
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import sqlite3
import time
import re
import warnings

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

ROOT = Path(__file__).resolve().parents[2]

RATINGS_PATH = ROOT / "data" / "processed" / "ml-10M100K" / "ratings_clean.dat"
MOVIES_PATH  = ROOT / "data" / "processed" / "ml-10M100K" / "movies.dat"
DEFAULT_DB_PATH = ROOT / "src" / "app" / "user_data.db"


# ---------------------------------------------------------
# Frontend helpers
# ---------------------------------------------------------

def get_frontend_ratings(db_path: str | Path) -> pd.DataFrame:
    """
    Pull only ratings of users that finished initial flow (done=1).
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
    Normalize titles so DB titles (no year) can match MovieLens titles (with year).
    """
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s*\(\d{4}\)\s*$", "", s)   # remove trailing (YYYY)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def load_movies(movies_path: str | Path) -> pd.DataFrame:
    """
    Load movies.dat (CSV): movie_id,title,genres (or fallback).
    """
    df = pd.read_csv(movies_path)
    if not {"movie_id", "title"}.issubset(df.columns):
        raise ValueError(f"{movies_path} must contain at least movie_id and title")
    df["title_norm"] = df["title"].map(_normalize_title)
    return df


def convert_frontend_to_movielens_schema(
    db_path: str | Path,
    movies_path: str | Path,
    base_max_user_id: int,
) -> pd.DataFrame:
    """
    Convert frontend DB ratings into MovieLens-like schema:
      username (string), user_id (numeric appended), movie_id, rating, timestamp

    IMPORTANT: We keep 'username' so we can later export username -> internal_user_id
    using the merged dataframe (source of truth).
    """
    db_ratings = get_frontend_ratings(db_path)
    if db_ratings.empty:
        return pd.DataFrame(columns=["username", "user_id", "movie_id", "rating", "timestamp"])

    movies_df = load_movies(movies_path)

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

    # Drop unmatched titles
    missing = merged["movie_id"].isna().sum()
    if missing:
        examples = merged.loc[merged["movie_id"].isna(), "movie"].dropna().unique()[:5]
        warnings.warn(
            f"{missing} DB ratings could not be mapped to movie_id. Dropping them. Examples: {examples}"
        )
        merged = merged.dropna(subset=["movie_id"]).copy()

    if merged.empty:
        return pd.DataFrame(columns=["username", "user_id", "movie_id", "rating", "timestamp"])

    # username -> numeric user_id appended after MovieLens max user_id (deterministic ordering)
    usernames = sorted(merged["username"].astype(str).unique())
    user_map = {u: base_max_user_id + 1 + i for i, u in enumerate(usernames)}
    merged["user_id"] = merged["username"].astype(str).map(user_map).astype("int64")

    # timestamp (DB has none): set to "now"
    ts = int(time.time())
    merged["timestamp"] = ts

    # Return with username preserved (MovieLens rows will have username = NaN after concat)
    df = pd.DataFrame({
        "username": merged["username"].astype(str),
        "user_id": merged["user_id"].astype("int64"),
        "movie_id": merged["movie_id"].astype("int64"),
        "rating": merged["rating"].astype("float32"),
        "timestamp": merged["timestamp"].astype("int64"),
    })

    df = df.sort_values("timestamp").drop_duplicates(["user_id", "movie_id"], keep="last")
    return df


# ---------------------------------------------------------
# MovieLens loading
# ---------------------------------------------------------

def load_ratings(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=",",
        header=0,
        dtype={
            "user_id": "Int64",
            "movie_id": "Int64",
            "rating": "float32",
            "timestamp": "Int64",
        },
        on_bad_lines="skip",
    )
    df = df.dropna(subset=["user_id", "movie_id", "rating", "timestamp"])
    return df


def build_interactions(df: pd.DataFrame, return_mappings: bool = False):
    # factorize ignores extra columns like "username"
    user_ids, user_unique = pd.factorize(df["user_id"])
    item_ids, item_unique = pd.factorize(df["movie_id"])

    interactions = Interactions(
        user_ids=user_ids.astype("int32"),
        item_ids=item_ids.astype("int32"),
        ratings=df["rating"].astype("float32").values,
        timestamps=df["timestamp"].astype("int64").values.astype("int32"),
        num_users=len(user_unique),
        num_items=len(item_unique),
    )

    if return_mappings:
        return interactions, user_unique, item_unique

    return interactions


# ---------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------

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
                ["user_id", "movie_id"], keep="last"
            )

    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=seed)

    interactions = build_interactions(df)

    rng_test = np.random.RandomState(seed)
    rng_val = np.random.RandomState(seed + 1)

    train_val, test = user_based_train_test_split(
        interactions, test_percentage=user_test_percentage, random_state=rng_test
    )

    train, val = user_based_train_test_split(
        train_val,
        test_percentage=user_val_percentage_from_train,
        random_state=rng_val
    )

    return train, val, test


def load_full_dataframe(
    ratings_path: str | Path = RATINGS_PATH,
    movies_path: str | Path = MOVIES_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
    include_frontend: bool = False,
) -> pd.DataFrame:
    """
    Return the FULL merged ratings dataframe.
    Includes a 'username' column for frontend rows (NaN for MovieLens rows).
    Includes a small sanity check print when include_frontend=True.
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
                ["user_id", "movie_id"], keep="last"
            )

        # ---- Sanity check ----
        if "username" in df.columns:
            n_front = int(df["username"].dropna().nunique())
            examples = df["username"].dropna().astype(str).unique()[:5].tolist()
        else:
            n_front = 0
            examples = []
        print("frontend usernames in full_df:", n_front)
        print("example frontend usernames:", examples)

    return df


def main() -> None:
    train, val, test = prepare_datasets()

    print("Training small implicit BPR factorization model to test data is formatted correctly...")
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
