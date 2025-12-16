# Recommender – Training and Usage Guide

This document explains how to train the recommender models, reload it, make predictions, and generate top-N recommendations.

---

## 1. How to Run the Training Script

### Standard training

```
python src/training/train_model.py
```

This will:

* Load MovieLens-10M ratings (`ratings_clean.dat`)
* Split the data into train, validation, and test sets
* Instantiate and train the Spotlight BPR recommender
* Evaluate it
* Save the trained model to `models/`
* Log metrics to MLflow


## 2. Expected Outputs and File Locations

### Model artifacts

* `models/bpr_recommender.pt`
* `models/top12_recommendations_with_titles.csv`

### MLflow tracking directory

* `mlruns/`

### Dataset (via DVC)

* `data/processed/ml-10M100K/ratings_clean.dat`
* `data/processed/ml-10M100K/movies.dat`

---

## 3. Generating Top-10 Recommendations

Generate recommendations for all users:

```
python src/training/generate_top10.py
```

This script:

1. Loads ratings + movie metadata
2. Reconstructs user/item ID mappings
3. Loads the trained recommender
4. Computes scores for every user
5. Extracts top-10 items
6. Maps item IDs → movieId → movie titles
7. Saves recommendations output file

### Output file

* Human-readable titles: `models/top12_recommendations_with_titles.csv`

---

## 4. Reloading a Saved Model

```python
import torch
recommender = torch.load("models/bpr_recommender.pt")
model = recommender.model
```

---

## 5. Making Predictions Programmatically

Example: get top-10 recommendations for one user.

```python
import torch
import numpy as np

recommender = torch.load("models/bpr_recommender.pt")
model = recommender.model

user_id = 42
scores = model.predict(user_id)

top10 = np.argsort(scores)[-10:][::-1]
print(top10)
```


---

## 6. Full Pipeline Workflow

### Step 1 — Pull dataset via DVC

```
dvc pull
```

### Step 2 — Train model

```
python src/training/train_model.py --model bpr --split user_eval_refit
```

### Step 3 — Generate top-10 recommendations

```
python src/training/generate_top10.py
```

### Step 4 — View output

Open:

```
models/top12_recommendations_with_titles.csv
```

````markdown


## 7. Model Comparison and Selection

This project supports **two recommender models**:

- **BPR Recommender**
- **Adaptive Recommender**

Both models share the same training and evaluation pipeline and can be compared in a reproducible way using MLflow.

---

### Training a Single Model

You can explicitly choose which model to train using the `--model` argument.

Train only the BPR recommender:

```bash
python src/training/train_model.py --model bpr
````

Train only the adaptive recommender:

```bash
python src/training/train_model.py --model adaptive
```

By default, this performs a standard train/validation/test split and logs metrics to MLflow.

---

### Final Training on All Data (No User Holdout)

When training a **final production model**, you should use the following split:

```bash
python src/training/train_model.py --model bpr --split user_eval_refit
```

This ensures that:

* All users and interactions are used for training
* No users are held out for evaluation
* Metrics are still logged because it trains on train-val-test first and then retrains on the full dataset
* The resulting model is suitable for deployment

⚠️ Use mode **only after** selecting the best model.

---

### Hyperparameter Search and Model Comparison

To compare models fairly, run a grid search across both recommenders:

```bash
python src/training/train_grid_search.py
```

This script:

* Runs hyperparameter searches for both **BPR** and **Adaptive** models
* Evaluates each configuration
* Logs parameters, metrics, and artifacts to **MLflow**
* Registers trained models in the MLflow model registry

---

### Selecting the Best Model

After experiments are logged in MLflow, you can automatically retrieve the best-performing model:

```bash
python src/training/select_best_model.py
```

This script:

* Queries the MLflow registry
* Selects the best currently registered model based on evaluation metrics
* Outputs the selected model

---

### Recommended Workflow

1. Run grid search:

   ```bash
   python src/training/train_grid_search.py
   ```

2. Inspect results in MLflow:

   ```bash
   mlflow ui
   ```

3. Select best model:

   ```bash
   python src/training/select_best_model.py
   ```

4. Retrain selected model on the full dataset:

   ```bash
   python src/training/train_model.py --model <best_model> --split user_eval_refit
   ```

---

```
```

---
