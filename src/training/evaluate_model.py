# src/training/evaluate_model.py

from pathlib import Path
import pandas as pd
import sqlite3
import torch
import mlflow
from spotlight.evaluation import mrr_score, precision_recall_score
from train_model import load_config, main as train_pipeline
import runpy

# ============================================================
# 0. Thresholds
# ============================================================

THRESHOLDS = {
    "test_mrr": 0.02
}

MRR_DROP_THRESHOLD = 0.10
NEW_USERS_PERCENT_THRESHOLD = 0.10


# ============================================================
# 1. DB Load
# ============================================================

def get_frontend_ratings(db_name: str):
    """
    Extract ratings only from frontend users who finished initial ratings.
    """
    conn = sqlite3.connect(db_name)
    query = """
        SELECT r.username, r.movie, r.rating
        FROM ratings r
        JOIN initial_ratings i ON r.username = i.username
        WHERE i.done = 1
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# ============================================================
# 2. Spotlight helper
# ============================================================

def prepare_interactions(df):
    """
    Map usernames and movies → integer IDs.
    """
    user_map = {u: idx for idx, u in enumerate(df["username"].unique())}
    item_map = {m: idx for idx, m in enumerate(df["movie"].unique())}

    df["user_id"] = df["username"].map(user_map)
    df["item_id"] = df["movie"].map(item_map)

    from spotlight.interactions import Interactions
    interactions = Interactions(
        user_ids=df["user_id"].values,
        item_ids=df["item_id"].values,
        ratings=df["rating"].values,
    )
    return interactions


# ============================================================
# 3. Evaluation
# ============================================================

def evaluate_frontend_model(model_path, db_name, project_root):
    """
    Compute model metrics on real frontend user ratings.
    """
    mlflow.set_tracking_uri(f"file:{project_root / 'mlruns'}")
    mlflow.set_experiment("evaluation_runs")

    recommender = torch.load(model_path, weights_only=False)

    df = get_frontend_ratings(db_name)
    if df.empty:
        print("No frontend ratings found.")
        return None

    interactions = prepare_interactions(df)

    test_mrr = mrr_score(recommender.model, interactions).mean()
    precision10, recall10 = precision_recall_score(recommender.model, interactions, k=10)

    results = {
        "test_mrr": float(test_mrr),
        "precision_at_10": float(precision10.mean()),
        "recall_at_10": float(recall10.mean()),
        "num_users": df["username"].nunique(),
        "num_ratings": len(df),
        "avg_ratings_per_user": len(df) / df["username"].nunique(),
    }

    print("Evaluation:", results)
    return results


# ============================================================
# 4. MLflow Logging (only once)
# ============================================================

def mlflow_log_eval(results):
    """
    Log the evaluation metrics.
    """
    with mlflow.start_run(run_name="periodic_evaluation"):
        for k, v in results.items():
            mlflow.log_metric(k, v)


def get_previous_mlflow_metrics():
    """
    Return the metrics from the *previous* run (second latest),
    to avoid comparing the run to itself.
    """
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("evaluation_runs")
    if exp is None:
        return None

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=2,
    )

    if len(runs) < 2:
        return None

    prev_run = runs[1]
    return prev_run.data.metrics


# ============================================================
# 5. Retraining Logic
# ============================================================

def should_retrain(current):
    retrain = False

    # Cold-start threshold
    if current.get("test_mrr", 0) < THRESHOLDS["test_mrr"]:
        print(f"[Trigger] test_mrr={current['test_mrr']:.4f} < {THRESHOLDS['test_mrr']:.4f}")
        retrain = True

    # Previous MLRuns
    prev = get_previous_mlflow_metrics()
    if not prev:
        return retrain

    # MRR DROP TRIGGER
    if "test_mrr" in prev:
        prev_mrr = prev["test_mrr"]
        if prev_mrr > 0 and current["test_mrr"] < prev_mrr * (1 - MRR_DROP_THRESHOLD):
            print(
                f"[Trigger] MRR dropped by more than {MRR_DROP_THRESHOLD*100:.0f}% "
                f"({current['test_mrr']:.4f} < {prev_mrr:.4f})"
            )
            retrain = True

    # NEW USER GROWTH TRIGGER
    if "num_users" in prev:
        prev_u = prev["num_users"]
        curr_u = current["num_users"]

        if curr_u > prev_u:
            growth = (curr_u - prev_u) / curr_u
            if growth > NEW_USERS_PERCENT_THRESHOLD:
                print(
                    f"[Trigger] New users increased by {growth:.2%} "
                    f"({prev_u} → {curr_u})"
                )
                retrain = True

    return retrain


# ============================================================
# 6. MAIN
# ============================================================

def main():
    config, project_root = load_config()

    db_name = str(project_root / "src" / "app" / "user_data.db")
    model_path = (
        Path(project_root)
        / config.get("paths", {}).get("models_dir", "models")
        / "bpr_recommender.pt"
    )

    # Step 1 — Evaluate
    curr_metrics = evaluate_frontend_model(model_path, db_name, project_root)
    if curr_metrics is None:
        print("No eval data. Exiting.")
        return

    # Step 2 — Log metrics
    mlflow_log_eval(curr_metrics)

    # Step 3 — Check triggers
    if should_retrain(curr_metrics):
        print("Retraining triggered.")
        #runpy.run_path("train_model.py", run_name="__main__")

    else:
        print("No retraining needed.")


if __name__ == "__main__":
    main()