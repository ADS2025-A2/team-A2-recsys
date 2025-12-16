# src/training/train_grid_search.py

import os

# Fix OpenMP / MKL issues on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import itertools
from pathlib import Path

import numpy as np
import yaml
import mlflow

from dvc.api import DVCFileSystem

from load import prepare_datasets
from baseline_recommender import SpotlightBPRRecommender
from adaptive_recommender import SpotlightAdaptiveRecommender
from spotlight.evaluation import mrr_score, precision_recall_score


# -------------------------------------------------------------------
# Config / paths helpers
# -------------------------------------------------------------------

def load_config() -> tuple[dict, Path]:
    """
    Load the YAML config from src/config/config.yaml and return
    (config_dict, project_root_path).
    """
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "src" / "config" / "config.yaml"

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    return config, project_root


def load_data_via_dvc(data_path: Path) -> None:
    """
    Ensure ratings file exists locally; try to pull via DVC if missing.
    """
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


# -------------------------------------------------------------------
# Hyperparameter grids
# -------------------------------------------------------------------

BPR_GRID = {
    "embedding_dim": [16, 32],
    "learning_rate": [5e-3, 1e-3],
    "num_negative_samples": [3, 5],
    # l2, n_iter, batch_size are set below
}

ADAPTIVE_GRID = {
    "embedding_dim": [16, 32],
    "learning_rate": [5e-3, 1e-3],
    "num_negative_samples": [3, 5],
    # "l2": [1e-6, 1e-5],
}


def grid_dict_to_param_list(grid: dict):
    """
    Turn {"a":[1,2], "b":[10,20]} into
    [{"a":1,"b":10}, {"a":1,"b":20}, {"a":2,"b":10}, {"a":2,"b":20}]
    """
    keys = list(grid.keys())
    values_lists = [grid[k] for k in keys]

    for combo in itertools.product(*values_lists):
        yield dict(zip(keys, combo))


# -------------------------------------------------------------------
# Model factory
# -------------------------------------------------------------------

def build_recommender(model_type: str, params: dict, seed: int = 42):
    """
    Create either a BPR or adaptive implicit recommender
    from params in the grid.
    """
    common_kwargs = {
        "embedding_dim": params.get("embedding_dim", 16),
        "n_iter": params.get("n_iter", 10),
        "batch_size": params.get("batch_size", 2048),
        "learning_rate": params.get("learning_rate", 5e-3),
        "l2": params.get("l2", 1e-6),
        "use_cuda": False,
        "random_state": np.random.RandomState(seed),
    }

    if model_type == "bpr":
        common_kwargs["num_negative_samples"] = params.get("num_negative_samples", 3)
        return SpotlightBPRRecommender(**common_kwargs)

    elif model_type == "adaptive":
        common_kwargs["num_negative_samples"] = params.get("num_negative_samples", 3)
        return SpotlightAdaptiveRecommender(**common_kwargs)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# -------------------------------------------------------------------
# Main grid-search procedure
# -------------------------------------------------------------------

def run_grid_search_for_model(
    model_type: str,
    param_grid: dict,
    train,
    val,
    experiment_name: str,
):
    """
    Run grid search for a single model type (bpr / adaptive).

    Returns:
        best_params, best_val_mrr
    """
    print(f"\n=== Grid search for model: {model_type} ===")

    best_mrr = -np.inf
    best_params = None

    for params in grid_dict_to_param_list(param_grid):
        print(f"\n--- {model_type} with params: {params} ---")

        with mlflow.start_run(run_name=f"{model_type}_grid_search"):
            # Log meta info
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("experiment", experiment_name)
            mlflow.log_params(params)

            # Build and train model
            recommender = build_recommender(model_type, params)
            recommender.model.fit(train, verbose=True)

            # Evaluate on validation set (for model selection)
            val_mrr = mrr_score(recommender.model, val, train=train).mean()
            val_prec10, val_rec10 = precision_recall_score(
                recommender.model, val, train=train, k=10
            )

            val_prec10 = float(val_prec10.mean())
            val_rec10 = float(val_rec10.mean())

            print(
                f"Validation MRR: {val_mrr:.4f} | "
                f"P@10: {val_prec10:.4f} | R@10: {val_rec10:.4f}"
            )

            mlflow.log_metric("val_mrr", float(val_mrr))
            mlflow.log_metric("val_precision_at_10", val_prec10)
            mlflow.log_metric("val_recall_at_10", val_rec10)

            # Track best config
            if val_mrr > best_mrr:
                best_mrr = val_mrr
                best_params = params

    print(
        f"\n>>> Best {model_type} config: {best_params} "
        f"with val MRR={best_mrr:.4f}"
    )
    return best_params, best_mrr


def main():
    # --------------------------------------------------------------
    # 0. Load config, paths, and prepare MLflow
    # --------------------------------------------------------------
    config, project_root = load_config()

    paths_cfg = config.get("paths", {})
    # Make sure this matches where the ratings actually are (raw or processed)
    data_rel = paths_cfg.get("data_path", "data/raw/ml-10M100K/ratings.dat")
    data_path = project_root / data_rel

    # Ensure data is present (DVC)
    load_data_via_dvc(data_path)

    training_cfg = config.get("training", {})
    seed = training_cfg.get("seed", 42)
    sample_size = training_cfg.get("sample_size", None)

    # MLflow
    tracking_uri = f"file:{(project_root / 'mlruns').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = config.get("experiment_name", "grid_search")
    mlflow.set_experiment(experiment_name)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment:   {experiment_name}")

    # --------------------------------------------------------------
    # 1. Prepare a SINGLE train/val/test split (shared by all runs)
    # --------------------------------------------------------------
    print("\nPreparing datasets (shared splits for all models/runs) ...")
    train, val, test = prepare_datasets(
        ratings_path=str(data_path),
        sample_size=sample_size,
        seed=seed,
    )

    print(f"Train interactions: {len(train)}")
    print(f"Val interactions:   {len(val)}")
    print(f"Test interactions:  {len(test)}")

    # --------------------------------------------------------------
    # 2. Grid search for each model type
    # --------------------------------------------------------------
    results = {}

    # BPR model
    best_bpr_params, best_bpr_mrr = run_grid_search_for_model(
        "bpr", BPR_GRID, train, val, experiment_name
    )
    results["bpr"] = {"best_params": best_bpr_params, "best_val_mrr": best_bpr_mrr}

    # adaptive implicit MF model
    best_adaptive_params, best_adaptive_mrr = run_grid_search_for_model(
        "adaptive", ADAPTIVE_GRID, train, val, experiment_name
    )
    results["adaptive"] = {
        "best_params": best_adaptive_params,
        "best_val_mrr": best_adaptive_mrr,
    }

    # --------------------------------------------------------------
    # 3. Print final comparison summary
    # --------------------------------------------------------------
    print("\n================ GRID SEARCH SUMMARY ================")
    for model_type, info in results.items():
        print(
            f"\nModel: {model_type}\n"
            f"  Best val MRR: {info['best_val_mrr']:.4f}\n"
            f"  Best params:  {info['best_params']}"
        )

    # Decide overall winner
    if results["bpr"]["best_val_mrr"] >= results["adaptive"]["best_val_mrr"]:
        winner = "bpr"
    else:
        winner = "adaptive"

    print(f"\n>>> Overall best model by val MRR: {winner}")
    print("====================================================\n")


if __name__ == "__main__":
    main()
