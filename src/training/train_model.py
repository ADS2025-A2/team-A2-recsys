# src/training/train_model_test.py

import yaml
from pathlib import Path
import mlflow
import os
import subprocess
import re
from github import Github
from dvc.api import DVCFileSystem
import argparse

# ==== IMPORT DATA AND MODEL PIPELINE ====
from load import prepare_datasets     
from spotlight.factorization.implicit import ImplicitFactorizationModel
from baseline_recommender import SpotlightBPRRecommender
from spotlight.evaluation import (
    mrr_score,
    precision_recall_score
)

# ============================================================
# CONFIG LOADING
# ============================================================

def load_config() -> dict:
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "src" / "config" / "config.yaml"
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    return config, project_root


# ============================================================
# DVC DATA LOADER
# ============================================================

def load_data_via_dvc(data_path: Path):
    """Ensures dataset exists locally; pulls via DVC if missing."""
    if data_path.exists():
        print(f"✔ Found data at: {data_path}")
        return

    print(f"Data not found at: {data_path}")
    print("Attempting: dvc pull ...")

    fs = DVCFileSystem(".")
    try:
        fs.get(str(data_path), str(data_path))
        print("✔ DVC pull successful.")
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to pull data via DVC.\nReason: {e}\n"
            f"Check DVC remotes via `dvc remote list`"
        )

    if not data_path.exists():
        raise FileNotFoundError(f"Data still not found after DVC pull: {data_path}")


# ============================================================
# GIT TAGGING
# ============================================================

def create_git_tag(version: str, repo_name: str, github_token: str):
    tag_name = f"1.{version}"
    try:
        subprocess.run(["git", "tag", tag_name], check=True)
        subprocess.run(["git", "push", "origin", tag_name], check=True)
        print(f"Created Git tag: {tag_name}")

        g = Github(github_token)
        repo = g.get_repo(repo_name)
        repo.create_git_release(
            tag=tag_name,
            name=f"Model_V{tag_name}",
            message="Automated release from MLflow pipeline"
        )
        print(f"GitHub release created for {tag_name}")

    except Exception as e:
        print(f"Git tagging failed: {e}")


class RecommenderWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, recommender):
        self.recommender = recommender

    def predict(self, context, model_input):
        # model_input is a DataFrame, return predictions as Series/array
        user_ids = model_input["user_id"].tolist()
        item_ids = model_input.get("item_id", None)
        return [self.recommender.predict_for_user(u, item_ids) for u in user_ids]

# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    # --------------------------------------------------------
    # Load config + MLflow setup
    # --------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run in fast mode (no training / registration / MLflow)")
    args = parser.parse_args()
    fast_mode = args.fast

    if fast_mode:
        print("===== FAST MODE ENABLED =====")
        load_data_via_dvc(data_path)
        print("Data is loaded via dvc. Skipping full training, evaluation, MLflow logging, and Git tagging.")

    config, project_root = load_config()

    tracking_uri = f"file:{(project_root / 'mlruns').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = config.get("experiment_name", "training_runs")
    mlflow.set_experiment(experiment_name)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment:   {experiment_name}")

    # Paths
    paths_cfg = config.get("paths", {})
    data_rel = paths_cfg.get("data_path", "ml-10M100K/ratings.dat")
    models_dir = project_root / paths_cfg.get("models_dir", "models")

    data_path = project_root / data_rel
    models_dir.mkdir(parents=True, exist_ok=True)

    if fast_mode:
        # Optionally, just load dataset and build model object
        train, val, test = prepare_datasets(
            ratings_path=str(data_path),
            sample_size=10,  # tiny sample to test pipeline
            seed=config["training"].get("seed", 42),
        )
        recommender = SpotlightBPRRecommender()
        print("FAST MODE: dummy dataset loaded, model instantiated")
        return  # Skip the rest
    
    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------
    with mlflow.start_run(run_name="spotlight_bpr_training") as run:

        run_id = run.info.run_id
        print(f"\n=== Starting MLflow Run: {run_id} ===")


        # 1) Load and split dataset (via load.py)

        print("Loading datasets via prepare_datasets() ...")

        train, val, test = prepare_datasets(
            ratings_path=str(data_path),
            sample_size=config["training"].get("sample_size", None),
            seed=config["training"].get("seed", 42),
        )

        # Log dataset size
        mlflow.log_metric("train_size", len(train))
        mlflow.log_metric("val_size", len(val))
        mlflow.log_metric("test_size", len(test))

        # 2) Build Spotlight BPR model

        print("Building BPR model ...")
        recommender = SpotlightBPRRecommender()

        # Log params to MLflow
        mlflow.log_params(recommender.params)

        # 3) Train model

        print("Training model ...")
        recommender.model.fit(train, verbose=True)


        # 4) Evaluate the model (evaluation.py)

        print("Evaluating model ...")

        mrr = mrr_score(recommender, test, train=train).mean()
        precision10, recall10 = precision_recall_score(recommender, test, train=train, k=10)

        mlflow.log_metric("MRR", float(mrr))
        mlflow.log_metric("Precision-10", float(precision10.mean()))
        mlflow.log_metric("Recall-10", float(recall10.mean()))

        print(f"MRR: {mrr:.4f}")
        print(f"Precision-10: {precision10.mean():.4f}")
        print(f"Recall-10: {recall10.mean():.4f}")


        # 5) Save / Log model to MLflow

        # Define model path
        model_path = models_dir / "baseline_recommender.pt"
        wrapped_model = RecommenderWrapper(recommender)

        mlflow.pyfunc.log_model(
            artifact_path="baseline_recommender",
            python_model=wrapped_model
        )

        # Register model (if needed)
        run_id = mlflow.active_run().info.run_id
        registered_model_name = config.get("registered_model_name", "baseline_recommender")
        model_uri = f"runs:/{run_id}/baseline_recommender"

        result = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name,
        )

        print(f"Registered new model version: {result.version}")

        # 6) Git tagging + GitHub release
        github_token = os.environ.get("GITHUB_TOKEN")
        repo_name = config.get("github_repo")  # "ADS2025-A2/team-A2-recsys"

        if github_token and repo_name:
            create_git_tag(str(result.version), repo_name, github_token)
        else:
            print("Skipping Git tag (missing repo_name or GITHUB_TOKEN)")


# ============================================================

if __name__ == "__main__":
    main()
