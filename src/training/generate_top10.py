# src/training/generate_top10.py
import os

# Fix OpenMP / MKL issues on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import csv
from pathlib import Path

import numpy as np
import torch
import yaml
import pandas as pd

from load import load_ratings, build_interactions
from baseline_recommender import SpotlightBPRRecommender


def load_movies(movies_path: Path) -> pd.DataFrame:
    """Load MovieLens movies.dat file."""
    cols = ["movie_id", "title", "genres"]
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=cols,
        encoding="latin-1",
    )
    return movies


# ---------------------------------------------------------
# Config loading
# ---------------------------------------------------------
def load_config() -> tuple[dict, Path]:
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "src" / "config" / "config.yaml"
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    return config, project_root


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------
def load_trained_recommender(model_path: Path) -> SpotlightBPRRecommender:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}.\n"
            f"Run the training script first to create baseline_recommender.pt."
        )

    print(f"Loading trained model from {model_path} ...")

    recommender = torch.load(model_path, map_location="cpu")
    return recommender


K = 12  # global K for convenience


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed" / "ml-10M100K" / "ratings_clean.dat"
    movies_path = project_root / "data" / "processed" / "ml-10M100K" / "movies.dat"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    # 1) Load ratings & interactions (with mappings)
    print(f"Loading ratings from {data_path} ...")
    df = load_ratings(str(data_path))
    interactions, user_mapping, item_mapping = build_interactions(
        df, return_mappings=True
    )

    num_users = interactions.num_users
    num_items = interactions.num_items
    print(f"Dataset has {num_users} users and {num_items} items.")

    # 2) Load trained model
    model_path = models_dir / "baseline_recommender.pt"
    print(f"Loading trained model from {model_path} ...")
    recommender: SpotlightBPRRecommender = torch.load(model_path, map_location="cpu")
    model = recommender.model

    # 3) Compute top-K internal item IDs per user (user-by-user, no batching)
    print("Computing top-K recommendations for all users (looping over users) ...")
    top_items_internal = np.zeros((num_users, K), dtype=np.int32)

    for u in range(num_users):
        # scores_u is a 1-D numpy array of length num_items
        scores_u = model.predict(u)

        # get indices of top-K scores
        topk_idx = np.argpartition(-scores_u, K)[:K]
        topk_idx = topk_idx[np.argsort(-scores_u[topk_idx])]

        top_items_internal[u] = topk_idx

        if (u + 1) % 1000 == 0:
            print(f"  processed {u + 1}/{num_users} users")

    # 4) Build mapping: internal item id -> original movieId -> title
    movies = load_movies(movies_path)

    # internal item id j  -> original movieId = item_mapping[j]
    internal_to_movieid = pd.Series(item_mapping, index=np.arange(len(item_mapping)))
    movieid_to_title = movies.set_index("movie_id")["title"]

    # internal id -> title (Series indexed by internal id)
    internal_to_title = internal_to_movieid.map(movieid_to_title).fillna(
        "UNKNOWN_TITLE"
    )

    # 5) Convert matrices to nice DataFrame with titles
    # user ids: internal -> original MovieLens userId
    user_ids_original = pd.Series(user_mapping, index=np.arange(len(user_mapping)))
    user_ids_col = user_ids_original.loc[np.arange(num_users)].reset_index(drop=True)

    # top_items_internal is (num_users, K) with internal ids
    top_items_df = pd.DataFrame(top_items_internal).applymap(
        lambda iid: internal_to_title[iid]
    )
    top_items_df.columns = [f"item_{i + 1}" for i in range(K)]

    result_df = pd.concat(
        [pd.Series(user_ids_col, name="user_id"), top_items_df],
        axis=1,
    )

    out_path = models_dir / "top10_recommendations_with_titles.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n✔ Saved recommendations with titles to {out_path}")


if __name__ == "__main__":
    main()
