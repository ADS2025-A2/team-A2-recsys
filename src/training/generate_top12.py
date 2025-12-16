# src/training/generate_top12.py
import os

# Fix OpenMP / MKL issues on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np
import torch
import pandas as pd
import spotlight

from load import build_interactions, load_full_dataframe
from baseline_recommender import SpotlightBPRRecommender


def load_movies(movies_path: Path) -> pd.DataFrame:
    """
    Load MovieLens movies file.
    Tries CSV first, falls back to MovieLens '::' format.
    """
    movies = pd.read_csv(movies_path)
    if {"movie_id", "title", "genres"}.issubset(movies.columns):
        return movies

    cols = ["movie_id", "title", "genres"]
    return pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=cols,
        encoding="latin-1",
    )


K = 12  # number of recommendations per user


def main():
    project_root = Path(__file__).resolve().parents[2]
    movies_path = project_root / "data" / "processed" / "ml-10M100K" / "movies.dat"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------
    # 1) Load full interaction data
    # ---------------------------------------------------------
    print("Loading ratings (MovieLens + frontend DB) ...")
    df = load_full_dataframe(include_frontend=True)

    interactions, user_mapping, item_mapping = build_interactions(
        df, return_mappings=True
    )
    csr = interactions.tocsr()

    # ---------------------------------------------------------
    # 2) Load trained model
    # ---------------------------------------------------------
    model_path = models_dir / "bpr_recommender.pt"
    print(f"Loading trained model from {model_path} ...")
    recommender: SpotlightBPRRecommender = torch.load(
        model_path, map_location="cpu", weights_only=False
    )
    model = recommender.model

    num_users_in_model = model._num_users
    num_users_total = interactions.num_users
    num_items = min(model._num_items, interactions.num_items)

    user_items_csr = csr[:num_users_in_model, :num_items]

    # ---------------------------------------------------------
    # 3) Build popular-item fallback pool
    # ---------------------------------------------------------
    print("Computing popular-item fallback list ...")
    item_popularity = np.asarray(
        csr[:, :num_items].getnnz(axis=0)
    ).ravel()
    popular_internal = np.argsort(-item_popularity)
    popular_pool = popular_internal[: max(1000, K * 50)]

    # ---------------------------------------------------------
    # 4) Generate recommendations
    # ---------------------------------------------------------
    print("Computing top-K recommendations ...")
    top_items_internal_all = np.full(
        (num_users_total, K), -1, dtype=np.int32
    )

    k_eff = min(K, max(1, num_items - 1))

    # Personalized users
    for u in range(num_users_in_model):
        scores = model.predict(u)[:num_items]

        seen = user_items_csr[u].indices
        if seen.size:
            scores[seen] = -np.inf

        if np.isfinite(scores).any():
            topk = np.argpartition(-scores, k_eff)[:k_eff]
            topk = topk[np.argsort(-scores[topk])]
        else:
            topk = np.array([], dtype=np.int32)

        if topk.size < K:
            pad = np.full(K - topk.size, -1, dtype=np.int32)
            top_items_internal_all[u] = np.concatenate([topk, pad])
        else:
            top_items_internal_all[u] = topk

        if (u + 1) % 1000 == 0:
            print(f"  processed {u + 1}/{num_users_in_model} users")

    # Fallback users
    if num_users_total > num_users_in_model:
        print(
            f"Found {num_users_total - num_users_in_model} extra users. "
            f"Using popular fallback recommendations."
        )

        for u in range(num_users_in_model, num_users_total):
            seen = set(csr[u, :num_items].indices.tolist())

            recs = []
            for iid in popular_pool:
                if iid not in seen:
                    recs.append(int(iid))
                    if len(recs) == K:
                        break

            if len(recs) < K:
                recs += [
                    int(i)
                    for i in popular_pool
                    if int(i) not in recs
                ][: K - len(recs)]

            top_items_internal_all[u] = np.array(recs[:K], dtype=np.int32)

    # ---------------------------------------------------------
    # 5) Map internal ids → titles
    # ---------------------------------------------------------
    movies = load_movies(movies_path)
    movies["movie_id"] = movies["movie_id"].astype(int)
    movieid_to_title = movies.set_index("movie_id")["title"]

    internal_to_movieid = pd.Series(
        item_mapping, index=np.arange(len(item_mapping))
    ).astype("Int64")
    internal_to_title = internal_to_movieid.map(
        movieid_to_title
    ).fillna("UNKNOWN_TITLE")

    # user_mapping maps internal_user_id -> original user_id (numeric)
    user_ids_original = pd.Series(
        user_mapping, index=np.arange(len(user_mapping))
    )
    user_ids_col = user_ids_original.loc[
        np.arange(num_users_total)
    ].reset_index(drop=True)

    def iid_to_title(iid: int) -> str:
        return "" if iid == -1 else str(
            internal_to_title.get(iid, "UNKNOWN_TITLE")
        )

    top_items_df = pd.DataFrame(top_items_internal_all).applymap(iid_to_title)
    top_items_df.columns = [f"item_{i + 1}" for i in range(K)]

    # ✅ NEW: internal_user_id column for stable lookup in frontend
    internal_user_ids_col = pd.Series(np.arange(num_users_total), name="internal_user_id")

    # Keep existing numeric user_id too (handy for debugging/legacy)
    result_df = pd.concat(
        [internal_user_ids_col, pd.Series(user_ids_col, name="user_id"), top_items_df],
        axis=1,
    )


    # ---------------------------------------------------------
    # 6) Save
    # ---------------------------------------------------------
    out_path = models_dir / "top12_recommendations_with_titles.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✔ Saved recommendations to {out_path}")


if __name__ == "__main__":
    main()
