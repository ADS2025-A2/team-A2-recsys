# 📁 Project Structure & Contribution Guidelines

This repository contains the full codebase, configuration, and documentation for our recommender system project.
Below you will find a clear explanation of the repository structure and **important rules all developers must follow when pushing new files or restructuring branches**.

---

## 📂 Repository Structure

```
.
├── .dvc/                     # DVC metadata
├── .github/                  # CI/CD workflows and automation
├── data/                     # Project datasets (tracked via DVC)
├── src/                      # Main source code
│   ├── app/                  # Inference application and API endpoints
│   ├── config/               # YAML config files for training/inference
│   ├── training/             # Training pipeline, model building, and utils
│   ├── __init__.py
│   ├── load.py               # Script to load models and data
│   ├── setup.py              # Project setup logic (temporary)
│   ├── test_app.py           # Test app script (for CI / smoke tests)
│   ├── train.py              # Training entry point
│   └── utils.py              # Helper functions
├── tests/                    # Unit tests (pytest)
├── .dvcignore                # Ignore file for DVC
├── .flake8                   # Flake8 linting config
├── .gitignore                # Files ignored by Git
├── dvc.yaml                  # DVC pipeline definition
├── params.yaml               # Parameters for DVC pipeline
├── pytest.ini                # pytest config
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
```

---

## What Goes Where?

### **`src/`**

All executable Python code **must** live inside `src/`.
This includes:

* Training scripts
* Model loading logic
* Utility functions
* Models that are part of the “chosen solution.”
* Config helpers
* App / API code

If you write code that will be imported from anywhere else, it **must** be inside `src/`.

---

### **`data/`**

* Only DVC-tracked datasets should appear here.
* Never push raw data to Git.
* Use `dvc add` and commit the metafiles.

---


### **`tests/`**

* Add all new unit tests here.
* Every function in `src/` that contains logic should eventually have a test.

---

### **Documentation Files**

* All `.md` documents belong at the project root unless they are module-specific (then place inside that module folder).

---

## Rules When Pushing to Branches

To maintain a clean history and functional CI/CD, follow the rules below:

1. Never commit untracked datasets or model artifacts

Allowed:

* DVC files (`.dvc`)
* Small test artifacts
  Forbidden:
* parquet, raw models, checkpoints
  Use:

```
dvc add data/myfile.csv
git add data/myfile.csv.dvc
```

---

2. All executable code must stay inside `src/`

Do **NOT** place scripts in root unless they are:

* Project entry points (`train_model.py`)
* Legacy migration scripts
* CI helper scripts

Otherwise, move them into:

* `src/app/` for application code
* `src/training/` for training logic
* `src/config/` for config loaders
* `src/utils.py` for small helpers

---

 3. Keep the branch structure intact

Example pitfalls:

❌ Adding a new folder at root without team approval
❌ Moving files out of `src/`
❌ Copying training scripts into new folders
❌ Adding multiple versions of the same script (e.g., `train2.py`, `train_new.py`)

---

4. Respect the Python formatting & linting rules

We enforce:

* Black formatting (line length: 88)
* Flake8 linting

5. Do not break the CI pipelines

CI checks:

* linting
* pytest
* DVC pull checks
* branch structure integrity

6. Commit messages must be descriptive

Examples:

Good:

* `Move load script into src`
* `Fix formatting in .flake8 ignore line`
* `Add placeholder function in utils.py`

Bad:

* `fix`
* `update stuff`
* `temp changes`

7. Avoid large structural changes inside feature branches**

If you modify:

* folder structure
* module imports
* config file layouts

→ inform the team in advance.

---

## Summary for Developers

When adding or modifying files:

* Put **all code inside `src/`**.
* Never commit raw data or models — use **DVC**.
* Keep file/folder names logical and consistent.
* Run **flake8 + black + pytest** before pushing.
* Avoid breaking import paths.
* Don’t create new top-level folders without approval.
* Add documentation for any new major feature.

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

# Data Loading

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
