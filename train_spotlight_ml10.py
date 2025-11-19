
"""
Train a Spotlight implicit BPR model on MovieLens-10M (ml-10M100K/ratings.dat).
"""

import numpy as np
import pandas as pd

from spotlight.interactions import Interactions
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import mrr_score, precision_recall_score
from spotlight.factorization.implicit import ImplicitFactorizationModel

RATINGS_PATH = "ml-10M100K/ratings.dat"
MODEL_OUT = "artifacts/spotlight_ml10m_bpr.pt"

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
        on_bad_lines="skip",   # <-- skip corrupt rows
    )

    # Drop any rows with missing values
    df = df.dropna(subset=["user_id", "item_id", "rating", "timestamp"])

    return df


def build_interactions(df):
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
        num_items=len(item_unique)
    )

    return interactions


def main() -> None:
    print(f"Loading ratings from {RATINGS_PATH} ...")
    df = load_ratings(RATINGS_PATH)
    print(f"Loaded {len(df):,} interactions")

    print("Building Spotlight Interactions object ...")
    interactions = build_interactions(df)

    print("Splitting into train and test sets ...")
    train, test = random_train_test_split(interactions, random_state=np.random.RandomState(42))

    print("Training implicit BPR factorization model ...")
    model = ImplicitFactorizationModel(
        loss="bpr",
        embedding_dim=64,
        n_iter=20,
        batch_size=1024,
        learning_rate=1e-3,
        l2=1e-6,
        num_negative_samples=5,
        use_cuda=False,  # set True if you have a working GPU+CUDA
        random_state=np.random.RandomState(42),
    )

    model.fit(train, verbose=True)

    print("Evaluating model ...")
    mrr = mrr_score(model, test, train=train).mean()
    precision, recall = precision_recall_score(model, test, train=train, k=10)
    print(f"MRR:            {mrr:.4f}")
    print(f"Precision@10:   {precision.mean():.4f}")
    print(f"Recall@10:      {recall.mean():.4f}")

    print(f"Saving model to {MODEL_OUT}")
    import torch
    import os

    os.makedirs("artifacts", exist_ok=True)
    torch.save(model, MODEL_OUT)
    print("Done.")

if __name__ == "__main__":
    main()


