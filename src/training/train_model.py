# src/training/train_model.py

import yaml
from pathlib import Path


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

    experiment_name = config.get("experiment_name", "unnamed_experiment")
    paths_cfg = config.get("paths", {})
    data_rel = paths_cfg.get("data_path", "ml-10M100K/ratings.dat")
    models_dir_rel = paths_cfg.get("models_dir", "models")

    data_path = project_root / data_rel
    models_dir = project_root / models_dir_rel

    print(f"Running experiment: {experiment_name}")
    print(f"Using data path: {data_path}")
    print(f"Models will be saved to: {models_dir}")

    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data from the DVC-tracked path
    load_data(data_path)

    # TODO: Data Scientist will implement:
    #  - feature engineering
    #  - train/validation split
    #  - model training
    #  - saving model artifact(s) into models_dir


if __name__ == "__main__":
    main()
