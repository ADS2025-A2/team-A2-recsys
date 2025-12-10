# src/training/evaluate_model.py

import os
from pathlib import Path
import pandas as pd
import torch
import mlflow
from spotlight.evaluation import mrr_score, precision_recall_score
from baseline_recommender import SpotlightBPRRecommender
from train_model import RecommenderWrapper, load_config, load_data_via_dvc
from spotlight.interactions import Interactions
import numpy as np

def interactions_from_tuples(tuples):
    user_ids = np.array([u for u, i, r in tuples])
    item_ids = np.array([i for u, i, r in tuples])
    ratings  = np.array([r for u, i, r in tuples])
    return Interactions(user_ids=user_ids, item_ids=item_ids, ratings=ratings)

# Metric thresholds
THRESHOLDS = {
    "test_mrr": 0.02,
    "precision_at_10": 0.4,
    "recall_at_10": 0.050,
}

def load_feedback_csv(csv_path: Path):
    """
    Load user feedback from CSV saved in DVC
    """
    if not csv_path.exists():
        load_data_via_dvc(csv_path)

    df = pd.read_csv(csv_path)

    # Map username/movie to integer IDs for Spotlight
    user_mapping = {}
    item_mapping = {}
    interactions = []

    user_counter = 0
    item_counter = 0

    for _, row in df.iterrows():
        username, movie, rating = row["username"], row["movie"], row["rating"]

        if username not in user_mapping:
            user_mapping[username] = user_counter
            user_counter += 1
        if movie not in item_mapping:
            item_mapping[movie] = item_counter
            item_counter += 1

        interactions.append((user_mapping[username], item_mapping[movie], rating))

    return interactions

def evaluate_model(model_path: Path, feedback_csv_path: Path, project_root: Path):
    mlflow.set_tracking_uri(f"file:{project_root / 'mlruns'}")
    mlflow.set_experiment("evaluation_runs")
    # Load trained model
    recommender: SpotlightBPRRecommender = torch.load(model_path, weights_only=False)

    interactions = load_feedback_csv(feedback_csv_path)
    interactions_obj = interactions_from_tuples(interactions) 
    
    test_mrr = mrr_score(recommender.model, interactions_obj).mean()
    precision10, recall10 = precision_recall_score(recommender.model, interactions_obj, k=10)


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


    with mlflow.start_run(run_name="periodic_evaluation"):
        mlflow.log_metrics(results)
        return results

if __name__ == "__main__":
    config, project_root = load_config()
    models_dir = project_root / config.get("paths", {}).get("models_dir", "models")
    model_path = models_dir / "baseline_recommender.pt"

    # Path to your CSV in DVC
    feedback_csv_path = project_root / "data" / "processed" / "ratings_from_db.csv"

    evaluate_model(model_path, feedback_csv_path, project_root)
