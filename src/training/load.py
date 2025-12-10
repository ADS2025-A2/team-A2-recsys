"""
Data loading & splitting utilities for MovieLens-10M.

Key function:
    prepare_datasets(...)
        -> returns (train, val, test) as Spotlight Interactions objects.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

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

RATINGS_PATH = ROOT / "data" / "raw" / "ml-10M100K" / "ratings.dat"


def load_ratings(path: str | Path) -> pd.DataFrame:
    """Load the MovieLens ratings.dat file into a pandas DataFrame."""
    cols = ["user_id", "item_id", "rating", "timestamp"]

    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=cols,
        dtype={
            "user_id": "Int64",
            "item_id": "Int64",
            "rating": "float32",
            "timestamp": "Int64",
        },
        on_bad_lines="skip",
    )

    # Drop any rows with missing values
    df = df.dropna(subset=["user_id", "item_id", "rating", "timestamp"])

    return df


def build_interactions(df: pd.DataFrame, return_mappings: bool = False):
    """
    Convert a ratings DataFrame into a Spotlight Interactions object.

    If return_mappings=True, also returns the arrays of original user and item IDs.
    """
    # Factorize gives integer-coded ids and mapping arrays
    user_ids, user_unique = pd.factorize(df["user_id"])
    item_ids, item_unique = pd.factorize(df["item_id"])

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
