# src/training/train_model.py

import yaml
from pathlib import Path
import mlflow
import os
from github import Github

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


from dvc.api import DVCFileSystem

def load_data(data_path: Path):
    """
    Try to load the dataset. If missing, automatically attempt a DVC pull.
    """

    if data_path.exists():
        print(f"Found data at: {data_path}")
    else:
        print(f"⚠️ Data file not found at: {data_path}")
        print("Attempting: dvc pull ...")

        # Create a DVC filesystem object (uses .dvc/config + remotes)
        fs = DVCFileSystem(".")

        try:
            # This pulls the file and its dependencies from the configured remote
            fs.get(str(data_path), str(data_path))
            print("✔ DVC pull successful.")
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to pull data via DVC.\n"
                f"Reason: {e}\n"
                f"Make sure DVC remotes are configured: `dvc remote list`"
            )

        # Re-check
        if not data_path.exists():
            raise FileNotFoundError(f"Data still not found after DVC pull: {data_path}")

    # Preview first lines
    print(f"Opening data file...")
    with data_path.open("r", encoding="latin-1") as f:
        for i in range(5):
            line = f.readline()
            if not line:
                break
            print(line.strip())


import subprocess
import re

class DummyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return ["dummy"] * len(model_input)


def create_git_tag(version: str, repo_name: str, github_token: str):
    """
    Creates a Git tag like 1.1, pushes it, and creates a GitHub release.
    """
    tag_name = f"1.{version}"
    try:
        # Create and push Git tag
        subprocess.run(["git", "tag", tag_name], check=True)
        subprocess.run(["git", "push", "origin", tag_name], check=True)
        print(f"Created and pushed Git tag: {tag_name}")

        # Create GitHub release
        g = Github(github_token)
        repo = g.get_repo(repo_name)
        repo.create_git_release(
            tag=tag_name,
            name=f"Release {tag_name}",
            message="Automated release from MLflow pipeline",
            draft=False,
            prerelease=False,
        )
        print(f"Created GitHub release for tag {tag_name}")

    except Exception as e:
        print(f"⚠️ Could not create Git tag / release: {e}")


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
    with mlflow.start_run(run_name="placeholder_training_run") as run:  #run_name should be changed when real training is implemented
        run_id = run.info.run_id
        print(f"Running experiment: {experiment_name}")
        print(f"Running experiment: {experiment_name}")
        print(f"Using data path: {data_path}")
        print(f"Models will be saved to: {models_dir}")

        models_dir.mkdir(parents=True, exist_ok=True)

        # Log placeholder parameters (for now)
        mlflow.log_param("data_path", str(data_path))
        mlflow.log_param("models_dir", str(models_dir))
        mlflow.log_param("model_status", "dummy_model")  # will change when real model is added
        mlflow.log_metric("placeholder_metric", 0.0) # Log a dummy metric (for now)

        # Load data from the DVC-tracked path (still just previewing)
        load_data(data_path)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=DummyModel()
        )

        # REGISTER MODEL → Automatically creates version

        model_uri = f"runs:/{run_id}/model"
        registered_model_name = "baseline_model"  # Change the model name when the real model will be registered. baseline_model is the dummy model that we use for initialization and testing. 

        result = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name,
        )
        
        version = result.version

        print(f"Registered new model version: {result.version}")

         # Git tagging & GitHub release
        github_token = os.environ.get("GITHUB_TOKEN")
        repo_name = config.get("github_repo")  # "ADS2025-A2/team-A2-recsys"
        if github_token and repo_name: # for this step, it is necessary to set a token by "export GITHUB_TOKEN", see README.md
            create_git_tag(str(result.version), repo_name, github_token)
        else:
            print("Skipping Git tag / GitHub release (GITHUB_TOKEN or repo_name not set)")



if __name__ == "__main__":
    main()
