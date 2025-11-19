# Code Style & Linting

We use Flake8 to enforce Python code quality and maintain readability. Please follow the guidelines below when developing.

### Key Points for Developers 
Line Length: 
Maximum of 88 characters per line.
Whitespace & Formatting:
W503 (line break before binary operator) is ignored — follow whichever style you prefer.
E305 (expected 2 blank lines) is ignored — minor deviations in spacing are acceptable.
Imports:
F401 (imported but unused) is ignored. Be mindful not to leave unnecessary imports in production code.
Code Complexity:
Functions or methods should not exceed complexity of 10. Refactor complex functions into smaller ones.
Exclusions:
Flake8 will not check files in .git, __pycache__, docs, venv, build, dist, mlruns, or .dvc.

# Training

Before running training, make sure you have pulled the data with DVC:

Please refer to `DVC_SETUP.md`

Then run the training script from the project root:
```
python train_model.py
```


# Continuous Integration (CI) Pipeline

This repository uses GitHub Actions to enforce code quality, data consistency, and reproducibility across the dev and main branches. The CI workflow is triggered on every push and pull request targeting either branch. The pipeline validates that the codebase, dependencies, dataset version (via DVC), and training workflow remain consistent with project standards.

### Pipeline Overview

1) The CI job executes on an isolated Ubuntu runner and performs the following operations sequentially: 
-Repository checkout 
-Downloads the repository contents into the CI runner using actions/checkout@v4.

2) Python environment setup Installs and configures Python 3.10 using actions/setup-python@v5.

3) Dependency installation: Installs all project requirements from requirements.txt and additional MLflow dependencies.

4) Dataset synchronization via DVC: Executes dvc pull to retrieve the dataset from the Azure Blob Storage remote. Authentication is handled through repository-level GitHub Secrets

5) Static analysis (linting): Runs flake8 over src/ and tests/ to ensure code quality and adherence to style guidelines. 
Note: This step is intended to be extended as the codebase grows.

6) Automated test suite executes all unit and integration tests using pytest.
Note: The current tests are placeholders and primarily verify that modules can be imported and scripts run without errors. Full unit and integration tests will be implemented in future development.

7) Training pipeline validation: Runs the training script in lightweight --fast mode using a local MLflow tracking directory.

The CI pipeline ensures that any modification introduced into dev or main adheres to the repository’s quality, reproducibility, and dataset-consistency requirements. This workflow integrates code validation, data synchronization, and model-pipeline verification into a unified automated process appropriate for ML-oriented development teams.