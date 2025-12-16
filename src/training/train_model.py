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
import pandas as pd
from github import Github
from dvc.api import DVCFileSystem
import torch

# ==== IMPORT DATA AND MODEL PIPELINE ====
from load import prepare_datasets, load_full_dataframe, build_interactions
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

def load_config() -> tuple[dict, Path]:
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
    tag_name = f"{model_key}-2.{version}"
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
    Convert numeric-looking strings in params into numbers.
    """
    clean = {}
    for k, v in params.items():
        if isinstance(v, str):
            try:
                clean[k] = int(v)
                continue
            except ValueError:
                pass
            try:
                clean[k] = float(v)
                continue
            except ValueError:
                pass
        clean[k] = v
    return clean

# ============================================================
# MAPPING EXPORTS (FOR STREAMLIT)
# ============================================================
def export_user_mappings(models_dir: Path, full_df: pd.DataFrame, user_mapping) -> None:
    import numpy as np
    import pandas as pd

    models_dir.mkdir(parents=True, exist_ok=True)

    # internal -> original numeric user_id
    internal_to_original = pd.Series(user_mapping, index=np.arange(len(user_mapping)))

    debug_path = models_dir / "internal_to_original_user_id.csv"
    internal_to_original.rename("original_user_id").reset_index().rename(
        columns={"index": "internal_user_id"}
    ).to_csv(debug_path, index=False, encoding="utf-8")
    print(f"✔ Saved internal→original user id map: {debug_path}")

    # original numeric user_id -> internal
    original_to_internal = pd.Series(
        internal_to_original.index.values, index=internal_to_original.values
    ).to_dict()

    # frontend users are the ones with a non-null username column
    if "username" not in full_df.columns:
        frontend_map_path = models_dir / "frontend_user_map.csv"
        pd.DataFrame(columns=["username", "internal_user_id"]).to_csv(frontend_map_path, index=False, encoding="utf-8")
        print(f"⚠ full_df has no username column. Wrote empty: {frontend_map_path}")
        return

    frontend_users = (
        full_df[["username", "user_id"]]
        .dropna(subset=["username"])
        .drop_duplicates()
        .copy()
    )
    frontend_users["username"] = frontend_users["username"].astype(str)
    frontend_users["user_id"] = frontend_users["user_id"].astype(int)

    frontend_users["internal_user_id"] = frontend_users["user_id"].map(original_to_internal)
    frontend_users = frontend_users.dropna(subset=["internal_user_id"]).copy()
    frontend_users["internal_user_id"] = frontend_users["internal_user_id"].astype(int)

    frontend_map_path = models_dir / "frontend_user_map.csv"
    frontend_users[["username", "internal_user_id"]].to_csv(frontend_map_path, index=False, encoding="utf-8")
    print(f"✔ Saved frontend username→internal_user_id map: {frontend_map_path} (rows={len(frontend_users)})")

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
        user_ids = model_input["user_id"].tolist()
        item_ids = model_input.get("item_id", None)
        return [self.recommender.predict_for_user(u, item_ids) for u in user_ids]

# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run in fast mode (no training / registration / MLflow)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Which model key to train: 'all' or a specific key (e.g., 'bpr', 'adaptive')",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="user_eval_refit",
        choices=["user", "none", "user_eval_refit"],
        help=(
            "Split strategy: "
            "'user' = train/eval on split (may miss users), "
            "'none' = train on ALL data (no eval), "
            "'user_eval_refit' = eval on split then refit on ALL data before saving."
        ),
    )

    args = parser.parse_args()
    fast_mode = args.fast
    selected_model = args.model
    split_mode = args.split

    # ---------- Load config & paths ----------
    config, project_root = load_config()

    paths_cfg = config.get("paths", {})
    data_rel = paths_cfg.get("data_path", "data/processed/ml-10M100K/ratings_clean.dat")
    models_dir_rel = paths_cfg.get("models_dir", "models")

    data_path = project_root / data_rel
    models_dir = project_root / models_dir_rel
    models_dir.mkdir(parents=True, exist_ok=True)

    # Ensure base ratings file exists locally
    load_data_via_dvc(data_path)

    training_cfg = config.get("training", {})
    sample_size = training_cfg.get("sample_size", None)
    seed = training_cfg.get("seed", training_cfg.get("random_seed", 42))

    # ---------- FAST MODE ----------
    if fast_mode:
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
    print(f"Split mode:          {split_mode}")
    print(f"Selected model:      {selected_model}")

    # ---------- Build MODELS config ----------
    models_cfg = config.get("models")
    if not models_cfg:
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
            "bpr": {"class_name": "SpotlightBPRRecommender", "params": bpr_params}
        }

    primary_model_key = config.get("primary_model_key", "bpr")

    # ---------------------------------------------------------
    # Data loading:
    # - split sets for evaluation (if needed)
    # - full interactions for "train on all" refit + mapping export
    # ---------------------------------------------------------
    train = val = test = None
    if split_mode in ("user", "user_eval_refit"):
        print("Loading split datasets via prepare_datasets(include_frontend=True) ...")
        train, val, test = prepare_datasets(
            include_frontend=True,
            ratings_path=str(data_path),
            sample_size=sample_size,
            seed=seed,
        )
        print(f"Train interactions: {len(train)}")
        print(f"Val interactions:   {len(val)}")
        print(f"Test interactions:  {len(test)}")

    # Full data (for serving / coverage + stable mapping)
    print("Loading FULL dataframe (MovieLens + frontend) ...")
    full_df = load_full_dataframe(
        ratings_path=str(data_path),
        include_frontend=True,
    )
    if sample_size is not None and sample_size < len(full_df):
        full_df = full_df.sample(sample_size, random_state=seed)

    full_interactions, user_mapping, item_mapping = build_interactions(
        full_df, return_mappings=True
    )
    print(
        f"Full interactions: {len(full_interactions)} "
        f"(users={full_interactions.num_users}, items={full_interactions.num_items})"
    )

    # Export stable mapping files for Streamlit
    export_user_mappings(models_dir=models_dir, full_df=full_df, user_mapping=user_mapping)

    # Which interactions do we train on initially?
    if split_mode == "none":
        initial_train_interactions = full_interactions
    else:
        initial_train_interactions = train

    # ---------------------------------------------------------
    # Train models
    # ---------------------------------------------------------
    for model_key, model_info in models_cfg.items():
        if selected_model != "all" and model_key != selected_model:
            continue

        class_name = model_info.get("class_name", "SpotlightBPRRecommender")
        params = coerce_numeric_params(dict(model_info.get("params", {})))
        params.setdefault("random_state", np.random.RandomState(seed))

        ModelClass = MODEL_REGISTRY.get(class_name)
        if ModelClass is None:
            raise ValueError(f"Unknown model class_name: {class_name}")

        run_name = f"{model_key}_{class_name}"

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            print(f"\n=== Starting MLflow Run for model '{model_key}': {run_id} ===")

            mlflow.set_tag("model_key", model_key)
            mlflow.set_tag("model_class", class_name)
            mlflow.set_tag("split_mode", split_mode)
            mlflow.log_params(params)

            # Basic dataset sizes
            mlflow.log_metric("full_size", float(len(full_interactions)))
            mlflow.log_metric("full_num_users", float(full_interactions.num_users))
            mlflow.log_metric("full_num_items", float(full_interactions.num_items))
            if train is not None:
                mlflow.log_metric("train_size", float(len(train)))
                mlflow.log_metric("val_size", float(len(val)))
                mlflow.log_metric("test_size", float(len(test)))

            # 1) Build model
            print(f"\n=== Building model '{model_key}' ({class_name}) ===")
            recommender = ModelClass(**params)

            # 2) Fit initial
            print("\n=== Training model ===")
            recommender.fit(initial_train_interactions, verbose=True)

            # 3) Evaluate (if we have splits)
            if split_mode in ("user", "user_eval_refit"):
                print("\n=== Evaluating model on validation and test sets ===")
                val_mrr = mrr_score(recommender.model, val, train=train).mean()
                mlflow.log_metric("val_mrr", float(val_mrr))
                print(f"Validation MRR: {val_mrr:.4f}")

                test_mrr = mrr_score(recommender.model, test, train=train).mean()
                precision10, recall10 = precision_recall_score(
                    recommender.model, test, train=train, k=10
                )
                mlflow.log_metric("test_mrr", float(test_mrr))
                mlflow.log_metric("precision_at_10", float(precision10.mean()))
                mlflow.log_metric("recall_at_10", float(recall10.mean()))

                print(f"Test MRR:          {test_mrr:.4f}")
                print(f"Test Precision@10: {precision10.mean():.4f}")
                print(f"Test Recall@10:    {recall10.mean():.4f}")

            # 4) Refit on ALL data (recommended for serving coverage)
            if split_mode == "user_eval_refit":
                print("\n=== Re-fitting model on FULL data for serving (all users/items) ===")
                recommender = ModelClass(**params)  # re-init cleanly
                recommender.fit(full_interactions, verbose=True)

            # 5) Save local model
            local_model_path = models_dir / f"{model_key}_recommender.pt"
            print(f"\nSaving trained model locally to {local_model_path} ...")
            torch.save(recommender, local_model_path)
            print("Local model save complete.")

            # 6) Log model to MLflow
            artifact_path = f"{model_key}_recommender"
            wrapped_model = RecommenderWrapper(recommender)
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=wrapped_model,
            )

            # 7) Register model
            registered_model_name = (
                model_info.get("registered_model_name")
                or config.get("registered_model_name")
                or f"spotlight_{model_key}_recommender"
            )
            model_uri = f"runs:/{run_id}/{artifact_path}"
            print(f"Registering model '{registered_model_name}' from URI: {model_uri}")
            result = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )
            print(f"Registered new model version: {result.version}")

            # 8) Git tagging (primary model only)
            github_token = os.environ.get("GITHUB_TOKEN")
            repo_name = "ADS2025-A2/team-A2-recsys"

            if model_key == primary_model_key:
                if github_token and repo_name:
                    create_git_tag(str(result.version), repo_name, github_token, model_key)
                else:
                    print(
                        "Skipping Git tag for primary model "
                        f"'{model_key}' (missing repo_name or GITHUB_TOKEN)."
                    )

    print("\n=== Training complete. ===")

if __name__ == "__main__":
    main()

