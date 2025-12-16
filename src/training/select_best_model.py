# src/training/select_best_model.py

"""
Automated model selection using MLflow.

Running this script will:
  - Look up an MLflow experiment (default from config.yaml)
  - Compare runs using a chosen metric (default: test_mrr)
  - Print:
        * best model type (model_key + model_class)
        * metric values
        * MLflow model URI
    so that team can easily retrieve and load the best model.

Example usage:

    python -m src.training.select_best_model
    python -m src.training.select_best_model --metric val_mrr
"""

import os

# Fix OpenMP / MKL issues on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from pathlib import Path
from typing import Tuple

import yaml
import mlflow
from mlflow.tracking import MlflowClient


# ------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------

def load_config() -> Tuple[dict, Path]:
    """
    Load the YAML config from src/config/config.yaml
    and return (config_dict, project_root_path).
    """
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "src" / "config" / "config.yaml"

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    return config, project_root


# ------------------------------------------------------------
# Core logic
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select best model run from MLflow experiment."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name. "
             "Defaults to experiment_name in config.yaml.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="test_mrr",
        help="Metric to optimize when selecting the best run "
             "(e.g. test_mrr, val_mrr, precision_at_10).",
    )
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        default=True,
        help="If set, higher metric is better (default). "
             "Use --no-higher-is-better for metrics like RMSE.",
    )
    # allow explicit false via --no-higher-is-better
    parser.add_argument(
        "--no-higher-is-better",
        dest="higher_is_better",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config + project root
    config, project_root = load_config()

    # MLflow tracking URI consistent with training/train_model.py
    tracking_uri = f"file:{(project_root / 'mlruns').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment_name = args.experiment_name or config.get(
        "experiment_name", "training_runs"
    )
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow.")

    print(f"Using experiment: {experiment_name} (id={experiment.experiment_id})")
    metric_name = args.metric

    # Get all finished runs in this experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
    )

    if not runs:
        raise RuntimeError(f"No finished runs found in experiment '{experiment_name}'.")

    best_run = None
    best_value = None

    for run in runs:
        metrics = run.data.metrics
        if metric_name not in metrics:
            # Skip runs that don't have the selected metric
            continue

        value = metrics[metric_name]

        if best_value is None:
            best_value = value
            best_run = run
            continue

        if args.higher_is_better and value > best_value:
            best_value = value
            best_run = run
        elif not args.higher_is_better and value < best_value:
            best_value = value
            best_run = run

    if best_run is None:
        raise RuntimeError(
            f"No runs contained metric '{metric_name}' in experiment '{experiment_name}'."
        )

    # Extract information about the best run
    run_id = best_run.info.run_id
    params = best_run.data.params
    metrics = best_run.data.metrics
    tags = best_run.data.tags

    model_key = tags.get("model_key", "<unknown>")
    model_class = tags.get("model_class", params.get("model_class", "<unknown>"))

    # Artifact path is "<model_key>_recommender", matching train_model.py
    artifact_path = f"{model_key}_recommender"
    model_uri = f"runs:/{run_id}/{artifact_path}"

    # --------------------------------------------------------
    # Pretty print summary
    # --------------------------------------------------------
    print("\n================ BEST MODEL SELECTION ================")
    print(f"Experiment name:    {experiment_name}")
    print(f"Selection metric:   {metric_name}")
    print(f"Higher is better:   {args.higher_is_better}")
    print(f"\nBest run id:        {run_id}")
    print(f"Best model key:     {model_key}")
    print(f"Best model class:   {model_class}")
    print(f"{metric_name}:      {best_value:.6f}")

    print("\nAll metrics for best run:")
    for k, v in sorted(metrics.items()):
        try:
            print(f"  {k}: {float(v):.6f}")
        except Exception:
            print(f"  {k}: {v}")

    print("\nMLflow model URI for best model:")
    print(f"  {model_uri}")
    print("\nYou can load this model in Python via:\n")
    print("  import mlflow")
    print(f"  model = mlflow.pyfunc.load_model('{model_uri}')")
    print("  # model.predict(df_with_user_ids)\n")
    print("======================================================\n")


if __name__ == "__main__":
    main()
