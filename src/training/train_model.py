# src/training/train_model.py

import os

# ---- Fix OpenMP / MKL issues on macOS ----
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import yaml
import mlflow
import argparse
import subprocess
from pathlib import Path

from github import Github
from dvc.api import DVCFileSystem

import torch

# ==== IMPORT DATA AND MODEL PIPELINE ====
from load import prepare_datasets
from baseline_recommender import SpotlightBPRRecommender
from spotlight.evaluation import mrr_score, precision_recall_score


# ============================================================
# CONFIG LOADING
# ============================================================

def load_config() -> dict:
    """
    Load the YAML config from src/config/config.yaml
    """
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "src" / "config" / "config.yaml"
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    return config, project_root


# ============================================================
# DVC DATA LOADER
# ============================================================

def load_data_via_dvc(data_path: Path):
    """Ensure dataset exists locally; pull via DVC if missing."""
    if data_path.exists():
        print(f"✔ Found data at: {data_path}")
        return

    print(f"Data not found at: {data_path}")
    print("Attempting: dvc pull via DVCFileSystem ...")

    fs = DVCFileSystem(".")

    try:
        fs.get(str(data_path), str(data_path))
        print("✔ DVC pull successful.")
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to pull data via DVC.\nReason: {e}\n"
            f"Check DVC remotes via `dvc remote list`."
        )

    if not data_path.exists():
        raise FileNotFoundError(f"Data still not found after DVC pull: {data_path}")


# ============================================================
# GIT TAGGING
# ============================================================

def create_git_tag(version: str, repo_name: str, github_token: str):
    """
    Create a git tag + GitHub release for the given model version.
    """
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
            message="Automated release from MLflow pipeline",
        )
        print(f"GitHub release created for {tag_name}")

    except Exception as e:
        print(f"Git tagging failed: {e}")


# ============================================================
# MLflow pyfunc wrapper
# ============================================================

class RecommenderWrapper(mlflow.pyfunc.PythonModel):
    """
    Wraps SpotlightBPRRecommender so it can be logged as an MLflow pyfunc model.
    """

    def __init__(self, recommender: SpotlightBPRRecommender):
        self.recommender = recommender

    def predict(self, context, model_input):
        """
        model_input: pandas.DataFrame with at least a 'user_id' column.
        """
        user_ids = model_input["user_id"].tolist()
        item_ids = model_input.get("item_id", None)

        # Return list of numpy arrays / scores per user
        return [self.recommender.predict_for_user(u, item_ids) for u in user_ids]


# MAIN TRAINING PIPELINE

def main():
    # ---------- CLI: fast mode ----------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run in fast mode (no training / registration / MLflow)",
    )
    args = parser.parse_args()
    fast_mode = args.fast

    # ---------- Load config & paths ----------
    config, project_root = load_config()

    paths_cfg = config.get("paths", {})
    data_rel = paths_cfg.get("data_path", "ml-10M100K/ratings.dat")
    models_dir_rel = paths_cfg.get("models_dir", "models")

    data_path = project_root / data_rel
    models_dir = project_root / models_dir_rel
    models_dir.mkdir(parents=True, exist_ok=True)

    # Ensure data exists 
    load_data_via_dvc(data_path)

    # ---------- FAST MODE ----------
    if fast_mode:
        # Just check dataset + build model; no training, no MLflow, no git.
        _ = SpotlightBPRRecommender()
        print(
            "FAST MODE: dataset verified via DVC, "
            "model instantiated. Skipping training, evaluation, MLflow, and Git tagging."
        )
        return

    # ---------- MLflow setup ----------
    tracking_uri = f"file:{(project_root / 'mlruns').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = config.get("experiment_name", "training_runs")
    mlflow.set_experiment(experiment_name)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment:   {experiment_name}")

    training_cfg = config.get("training", {})
    sample_size = training_cfg.get("sample_size", None)
    seed = training_cfg.get("seed", 42)

    # Hyperparameters for Spotlight BPR, read from config
    bpr_params = {
        "embedding_dim": training_cfg.get("embedding_dim", 16),
        "n_iter":        training_cfg.get("n_iter", 10),
        "batch_size":    training_cfg.get("batch_size", 2048),
        "learning_rate": training_cfg.get("learning_rate", 5e-3),
        "l2":            training_cfg.get("l2", 1e-6),
        "num_negative_samples": training_cfg.get("num_negative_samples", 3),
        "use_cuda":      training_cfg.get("use_cuda", False),
    }

    # ---------- Start MLflow run ----------
    with mlflow.start_run(run_name="spotlight_bpr_training") as run:
        run_id = run.info.run_id
        print(f"\n=== Starting MLflow Run: {run_id} ===")

        # 1) Load and split dataset
        print("Loading datasets via prepare_datasets() ...")
        train, val, test = prepare_datasets(
            ratings_path=str(data_path),
            sample_size=sample_size,
            seed=seed,
        )

        print(f"Train interactions: {len(train)}")
        print(f"Val interactions:   {len(val)}")
        print(f"Test interactions:  {len(test)}")

        mlflow.log_metric("train_size", len(train))
        mlflow.log_metric("val_size", len(val))
        mlflow.log_metric("test_size", len(test))

        # 2) Build baseline recommender with config hyper-params
        print("\n=== Building Spotlight BPR model ===")
        recommender = SpotlightBPRRecommender(**bpr_params)

        # Log hyper-parameters to MLflow
        mlflow.log_params(bpr_params)

        # 3) Train model
        print("\n=== Training model ===")
        recommender.model.fit(train, verbose=True)

        # 4) Validation metric (on val set) + final test metrics
        print("\n=== Evaluating model on validation and test sets ===")

        # validation metric
        val_mrr = mrr_score(recommender.model, val, train=train).mean()
        mlflow.log_metric("val_mrr", float(val_mrr))
        print(f"Validation MRR: {val_mrr:.4f}")

        # test metrics
        test_mrr = mrr_score(recommender.model, test, train=train).mean()
        precision10, recall10 = precision_recall_score(
            recommender.model, test, train=train, k=10
        )

        prec10_mean = float(precision10.mean())
        rec10_mean = float(recall10.mean())

        print(f"Test MRR:          {test_mrr:.4f}")
        print(f"Test Precision@10: {prec10_mean:.4f}")
        print(f"Test Recall@10:    {rec10_mean:.4f}")

        mlflow.log_metric("test_mrr", float(test_mrr))
        mlflow.log_metric("precision_at_10", prec10_mean)
        mlflow.log_metric("recall_at_10", rec10_mean)

        # 5) Save local .pt model
        local_model_path = models_dir / "baseline_recommender.pt"
        print(f"\nSaving trained model locally to {local_model_path} ...")
        torch.save(recommender, local_model_path)
        print("Local model save complete.")

        # 6) Log MLflow pyfunc model 
        print("\n=== Logging model to MLflow ===")
        wrapped_model = RecommenderWrapper(recommender)

        mlflow.pyfunc.log_model(
            artifact_path="baseline_recommender",
            python_model=wrapped_model,
        )

        # 7) Register model in MLflow Model Registry
        active_run = mlflow.active_run()
        active_run_id = active_run.info.run_id if active_run is not None else run_id

        registered_model_name = config.get(
            "registered_model_name", "baseline_recommender"
        )
        model_uri = f"runs:/{active_run_id}/baseline_recommender"

        print(f"Registering model '{registered_model_name}' from URI: {model_uri}")
        result = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name,
        )

        print(f"Registered new model version: {result.version}")

        # 8) Git tagging + GitHub release
        github_token = os.environ.get("GITHUB_TOKEN")
        repo_name = config.get(
            "github_repo",  # make sure key exists in config.yaml?
            "ADS2025-A2/team-A2-recsys",
        )

        if github_token and repo_name:
            create_git_tag(str(result.version), repo_name, github_token)
        else:
            print("Skipping Git tag (missing repo_name or GITHUB_TOKEN).")


# ============================================================

if __name__ == "__main__":
    main()
