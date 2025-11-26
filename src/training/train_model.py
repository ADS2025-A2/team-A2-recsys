# src/training/train_model.py

import yaml
from pathlib import Path
import mlflow

def load_config() -> dict:
    """
    Load the YAML config from src/config/config.yaml
    assuming this file is run from within the project root.
    """
    project_root = Path(__file__).resolve().parents[2]  # .../src/training -> .../src -> project root
    config_path = project_root / "src" / "config" / "config.yaml"
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    return config, project_root


def load_data(data_path: Path):
    """
    Placeholder data loader. For now, just check that the file exists
    and maybe read a few lines. Later, the DS can replace this with
    proper pandas/pyarrow loading etc.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    print(f"Found data at: {data_path}")
    # Example: preview first few lines
    with data_path.open("r", encoding="latin-1") as f:
        for i in range(5):
            line = f.readline()
            if not line:
                break
            print(line.strip())


def main():
    config, project_root = load_config()

    # MLflow tracking configuration (Task 1)
    tracking_uri = f"file:{(project_root / 'mlruns').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI set to: {tracking_uri}")


    # MLflow experiment setup (Task 2) 
    experiment_name = config.get("experiment_name", "training_runs")
    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment set to: {experiment_name}")
    

    paths_cfg = config.get("paths", {})
    data_rel = paths_cfg.get("data_path", "ml-10M100K/ratings.dat")
    models_dir_rel = paths_cfg.get("models_dir", "models")

    data_path = project_root / data_rel
    models_dir = project_root / models_dir_rel

    # MLflow run + placeholder logging (Task 3) 
    # This creates an MLflow run every time you call ⁠ python train_model.py ⁠
    with mlflow.start_run(run_name="placeholder_training_run") as run:
        run_id = run.info.run_id
        print(f"Running experiment: {experiment_name}")
        print(f"Running experiment: {experiment_name}")
        print(f"Using data path: {data_path}")
        print(f"Models will be saved to: {models_dir}")

        models_dir.mkdir(parents=True, exist_ok=True)

        # Log placeholder parameters (for now)
        mlflow.log_param("data_path", str(data_path))
        mlflow.log_param("models_dir", str(models_dir))
        mlflow.log_param("model_status", "no_model_yet")  # will change when real model is added
        mlflow.log_metric("placeholder_metric", 0.0) # Log a dummy metric (for now)

        # Load data from the DVC-tracked path (still just previewing)
        load_data(data_path)

        model_file = models_dir / "dummy_model.txt"
        model_file.write_text("This is a placeholder model.\n")

        # Log file as MLflow artifact
        mlflow.log_artifact(str(model_file), artifact_path="model")

        # REGISTER MODEL → Automatically creates version

        model_uri = f"runs:/{run_id}/model"
        registered_model_name = "baseline_model"

        result = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name,
        )

        print(f"Registered new model version: {result.version}")
  


if __name__ == "__main__":
    main()