"""
Train a Spotlight implicit BPR model on MovieLens-10M (ml-10M100K/ratings.dat),
with a reusable function to prepare data for training.
"""

import os

# Fix OpenMP / MKL issues on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

import numpy as np
import pandas as pd
from pathlib import Path

from spotlight.interactions import Interactions
from spotlight.cross_validation import user_based_train_test_split
from spotlight.evaluation import mrr_score, precision_recall_score
from spotlight.factorization.implicit import ImplicitFactorizationModel


ROOT = Path(__file__).resolve().parents[2]

RATINGS_PATH = ROOT / "data" / "processed" / "ml-10M100K" / "ratings_clean.dat"


def load_ratings(path: str) -> pd.DataFrame:
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
            "timestamp": "Int64"
        },
        on_bad_lines="skip",  
    )

    # Drop any rows with missing values
    df = df.dropna(subset=["user_id", "item_id", "rating", "timestamp"])

    return df


def build_interactions(df, return_mappings: bool = False):
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
    ratings_path: str = RATINGS_PATH,
    sample_size: int | None = None,
    user_test_percentage: float = 0.10,
    user_val_percentage_from_train: float = 0.111,
    seed: int = 42,
):
    """
    Load MovieLens ratings, optionally subsample, build Interactions,
    and return (train, val, test) user-based splits.

    sample_size=None means: use the full dataset.
    """
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
    # Small test to see if data can be used in training
    train, val, test = prepare_datasets()

    print("Training small implicit BPR factorization model to test data is formatted correctly...")
    model = ImplicitFactorizationModel(
        loss="bpr",
        embedding_dim=16,
        n_iter=10,
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
