# src/training/evaluate_frontend.py

from pathlib import Path
import pandas as pd
import numpy as np
import torch
import mlflow
from spotlight.interactions import Interactions
from spotlight.evaluation import mrr_score, precision_recall_score
from train_model import load_config
from baseline_recommender import SpotlightBPRRecommender
import sqlite3

# --- Helper to get frontend ratings ---
def get_frontend_ratings():
    """
    Extract ratings only from frontend users who finished initial ratings
    """
    conn = sqlite3.connect(DB_NAME)
    query = """
        SELECT r.username, r.movie, r.rating
        FROM ratings r
        JOIN initial_ratings i ON r.username = i.username
        WHERE i.done = 1
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# --- Convert dataframe to Spotlight Interactions ---
def interactions_from_df(df):
    """
    Map usernames and movies to integer IDs for Spotlight
    """
    user_mapping = {u: idx for idx, u in enumerate(df["username"].unique())}
    item_mapping = {i: idx for idx, i in enumerate(df["movie"].unique())}

    user_ids = df["username"].map(user_mapping).to_numpy()
    item_ids = df["movie"].map(item_mapping).to_numpy()
    ratings = df["rating"].to_numpy()

    return Interactions(user_ids=user_ids, item_ids=item_ids, ratings=ratings)


# --- Metric thresholds ---
THRESHOLDS = {
    "test_mrr": 0.02,
    "precision_at_10": 0.03,
    "recall_at_10": 0.03,
}


# --- Evaluation ---
def evaluate_frontend(model_path: Path, project_root: Path):
    mlflow.set_tracking_uri(f"file:{project_root / 'mlruns'}")
    mlflow.set_experiment("evaluation_runs")

    # Load model
    recommender: SpotlightBPRRecommender = torch.load(model_path, weights_only=False)

    # Load frontend ratings
    df = get_frontend_ratings()
    if df.empty:
        print("No frontend ratings found for evaluation.")
        return {}

    interactions = interactions_from_df(df)

    # Compute metrics
    test_mrr = mrr_score(recommender.model, interactions).mean()
    precision10, recall10 = precision_recall_score(recommender.model, interactions, k=10)

    results = {
        "test_mrr": float(test_mrr),
        "precision_at_10": float(precision10.mean()),
        "recall_at_10": float(recall10.mean()),
    }

    print("Evaluation results:", results)

    # Check thresholds
    for metric, threshold in THRESHOLDS.items():
        if results[metric] < threshold:
            print(f"Metric '{metric}' below threshold ({results[metric]:.4f} < {threshold})")

    # Log to MLflow
    with mlflow.start_run(run_name="periodic_evaluation"):
        mlflow.log_metrics(results)

    return results


if __name__ == "__main__":
    config, project_root = load_config()
    DB_NAME = project_root / "src" / "app" / "user_data.db"
    DB_NAME = str(DB_NAME) 
    models_dir = project_root / config.get("paths", {}).get("models_dir", "models")
    model_path = models_dir / "bpr_recommender.pt"

    evaluate_frontend(model_path, project_root)
