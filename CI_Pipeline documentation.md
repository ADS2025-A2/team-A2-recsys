Continuous Integration (CI) Pipeline Documentation

This repository uses GitHub Actions to enforce code quality, data consistency, and reproducibility across the dev and main branches. The CI workflow is triggered on every push and pull request targeting either branch. The pipeline validates that the codebase, dependencies, dataset version (via DVC), and training workflow remain consistent with project standards.

Pipeline Overview
1) The CI job executes on an isolated Ubuntu runner and performs the following operations sequentially:
Repository checkout
Downloads the repository contents into the CI runner using actions/checkout@v4.

2)Python environment setup
Installs and configures Python 3.10 using actions/setup-python@v5.

3) Dependency installation
Installs all project requirements from requirements.txt and additional MLflow dependencies.

4) Dataset synchronization via DVC
Executes dvc pull to retrieve the dataset from the Azure Blob Storage remote.
Authentication is handled through repository-level GitHub Secrets

5) Static analysis (linting)
Runs flake8 over src/ and tests/ to ensure code quality and adherence to style guidelines.
Note: This step is intended to be extended as the codebase grows.

7) Automated test suite
Executes all unit and integration tests using pytest.
Note: The current tests are placeholders and primarily verify that modules can be imported and scripts run without errors. Full unit and integration tests will be implemented in future development.

8) Training pipeline validation
Runs the training script in lightweight --fast mode using a local MLflow tracking directory.


The CI pipeline ensures that any modification introduced into dev or main adheres to the repository’s quality, reproducibility, and dataset-consistency requirements. This workflow integrates code validation, data synchronization, and model-pipeline verification into a unified automated process appropriate for ML-oriented development teams.
