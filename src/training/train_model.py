# src/training/train_model.py

import os

# ---- Fix OpenMP / MKL issues on macOS ----
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import yaml
import mlflow
import argparse
import subprocess
import numpy as np
from pathlib import Path

from github import Github
from dvc.api import DVCFileSystem

import torch

# ==== IMPORT DATA AND MODEL PIPELINE ====
from load import prepare_datasets
from baseline_recommender import SpotlightBPRRecommender
from adaptive_recommender import SpotlightAdaptiveRecommender

from spotlight.evaluation import mrr_score, precision_recall_score


# Registry so we can build models from config
MODEL_REGISTRY = {
    "SpotlightBPRRecommender": SpotlightBPRRecommender,
    "SpotlightAdaptiveRecommender": SpotlightAdaptiveRecommender,
}


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

def create_git_tag(version: str, repo_name: str, github_token: str, model_key: str):
    """
    Create a git tag + GitHub release for the given model version.

    Tag format is <model_key>-1.<version>, e.g. "bpr-1.3".
    """
    tag_name = f"{model_key}-1.{version}"
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
# PARAM SANITIZER
# ============================================================

def coerce_numeric_params(params: dict) -> dict:
    """
    Ensure that any numeric-looking strings in params are converted
    to actual Python numbers (ints / floats).

    This protects against config / logging quirks where values like
    "0.001" or "1e-6" might be strings instead of numbers.
    """
    clean = {}
    for k, v in params.items():
        if isinstance(v, str):
            # Try int
            try:
                iv = int(v)
                clean[k] = iv
                continue
            except ValueError:
                pass
            # Try float
            try:
                fv = float(v)
                clean[k] = fv
                continue
            except ValueError:
                pass
        clean[k] = v
    return clean


# ============================================================
# MLflow pyfunc wrapper
# ============================================================

class RecommenderWrapper(mlflow.pyfunc.PythonModel):
    """
    Generic MLflow pyfunc wrapper for our recommender models.

    Expects the wrapped object to expose `predict_for_user(user_id, item_ids=None)`.
    """

    def __init__(self, recommender):
        self.recommender = recommender

    def predict(self, context, model_input):
        """
        model_input: pandas.DataFrame with at least a 'user_id' column.
                     Optionally an 'item_id' column for scoring a subset.
        """
        user_ids = model_input["user_id"].tolist()
        item_ids = model_input.get("item_id", None)

        # Return list of numpy arrays / scores per user
        return [self.recommender.predict_for_user(u, item_ids) for u in user_ids]


# ============================================================
# MAIN TRAINING PIPELINE (MULTI-MODEL)
# ============================================================

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
    data_rel = paths_cfg.get("data_path", "data/processed/ml-10M100K/ratings_clean.dat")
    models_dir_rel = paths_cfg.get("models_dir", "models")

    data_path = project_root / data_rel
    models_dir = project_root / models_dir_rel
    models_dir.mkdir(parents=True, exist_ok=True)

    # Ensure data exists
    load_data_via_dvc(data_path)

    training_cfg = config.get("training", {})
    sample_size = training_cfg.get("sample_size", None)
    # Support both 'seed' and 'random_seed' keys
    seed = training_cfg.get("seed", training_cfg.get("random_seed", 42))

    # ---------- FAST MODE ----------
    if fast_mode:
        # Just check dataset + build a default model; no training, no MLflow, no git.
        _ = SpotlightBPRRecommender()
        print(
            "FAST MODE: dataset verified via DVC, "
            "baseline model instantiated. "
            "Skipping training, evaluation, MLflow, and Git tagging."
        )
        return

    # ---------- MLflow setup ----------
    tracking_uri = f"file:{(project_root / 'mlruns').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = config.get("experiment_name", "training_runs")
    mlflow.set_experiment(experiment_name)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment:   {experiment_name}")

    # ---------- Build MODELS config ----------
    #
    # These params can be set to the best hyperparameters found by the grid search.
    #
    # If `models` is missing, we fall back to a single BPR model
    # --------------------------------------------------------------
    models_cfg = config.get("models")

    if not models_cfg:
        # Backwards-compatible default: single BPR model with old training config
        bpr_params = {
            "embedding_dim": training_cfg.get("embedding_dim", 16),
            "n_iter": training_cfg.get("n_iter", 10),
            "batch_size": training_cfg.get("batch_size", 2048),
            "learning_rate": training_cfg.get("learning_rate", 5e-3),
            "l2": training_cfg.get("l2", 1e-6),
            "num_negative_samples": training_cfg.get("num_negative_samples", 3),
            "use_cuda": training_cfg.get("use_cuda", False),
        }
        models_cfg = {
            "bpr": {
                "class_name": "SpotlightBPRRecommender",
                "params": bpr_params,
            }
        }

    # Which model should trigger Git tagging by default
    primary_model_key = config.get("primary_model_key", "bpr")

    # ---------- Load and split dataset ONCE (shared splits) ----------
    print("Loading datasets via prepare_datasets() ...")
    train, val, test = prepare_datasets(
        ratings_path=str(data_path),
        sample_size=sample_size,
        seed=seed,
    )

    print(f"Train interactions: {len(train)}")
    print(f"Val interactions:   {len(val)}")
    print(f"Test interactions:  {len(test)}")

    # ---------- Train each model in turn ----------
    for model_key, model_info in models_cfg.items():
        class_name = model_info.get("class_name", "SpotlightBPRRecommender")
        params = dict(model_info.get("params", {}))  # shallow copy

        # Convert numeric-looking strings -> real numbers (protects against TypeError)
        params = coerce_numeric_params(params)

        # Ensure reproducible seed is passed into the model
        params.setdefault("random_state", np.random.RandomState(seed))

        ModelClass = MODEL_REGISTRY.get(class_name)
        if ModelClass is None:
            raise ValueError(f"Unknown model class_name: {class_name}")

        run_name = f"{model_key}_{class_name}"

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            print(f"\n=== Starting MLflow Run for model '{model_key}': {run_id} ===")

            # 1) Log dataset sizes (same for all models)
            mlflow.log_metric("train_size", len(train))
            mlflow.log_metric("val_size", len(val))
            mlflow.log_metric("test_size", len(test))

            # 2) Log meta info & hyperparameters
            mlflow.set_tag("model_key", model_key)
            mlflow.set_tag("model_class", class_name)

            mlflow.log_params(params)

            # 3) Build model
            print(f"\n=== Building model '{model_key}' ({class_name}) ===")
            recommender = ModelClass(**params)

            # 4) Train model
            print("\n=== Training model ===")
            recommender.fit(train, verbose=True)

            # 5) Validation + test metrics
            print("\n=== Evaluating model on validation and test sets ===")

            # Validation
            val_mrr = mrr_score(recommender.model, val, train=train).mean()
            mlflow.log_metric("val_mrr", float(val_mrr))
            print(f"Validation MRR: {val_mrr:.4f}")

            # Test
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

            # 6) Save local .pt model
            local_model_path = models_dir / f"{model_key}_recommender.pt"
            print(f"\nSaving trained model locally to {local_model_path} ...")
            torch.save(recommender, local_model_path)
            print("Local model save complete.")

            # 7) Log MLflow pyfunc model
            print("\n=== Logging model to MLflow ===")
            artifact_path = f"{model_key}_recommender"
            wrapped_model = RecommenderWrapper(recommender)

            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=wrapped_model,
            )

            # 8) Register model in MLflow Model Registry
            registered_model_name = (
                model_info.get("registered_model_name")
                or config.get("registered_model_name")
                or f"spotlight_{model_key}_recommender"
            )
            model_uri = f"runs:/{run_id}/{artifact_path}"

            print(
                f"Registering model '{registered_model_name}' "
                f"from URI: {model_uri}"
            )
            result = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )

            print(f"Registered new model version: {result.version}")

            # 9) Git tagging + GitHub release (only for primary model)
            github_token = os.environ.get("GITHUB_TOKEN")
            repo_name = config.get(
                "github_repo"
            )

            if model_key == primary_model_key:
                if github_token and repo_name:
                    create_git_tag(str(result.version), repo_name, github_token, model_key)
                else:
                    print(
                        "Skipping Git tag for primary model "
                        f"'{model_key}' (missing repo_name or GITHUB_TOKEN)."
                    )

    print("\n=== All configured models have been trained and logged. ===")


# ============================================================

if __name__ == "__main__":
    main()
