# Baseline Recommender – Training and Usage Guide

This document explains how to train the Spotlight BPR recommender model, reload it, make predictions, and generate top-N recommendations.

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

* `models/baseline_recommender.pt`
* `models/top10_recommendations_with_titles.csv`

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

* Human-readable titles: `models/top10_recommendations_with_titles.csv`

---

## 4. Reloading a Saved Model

```python
import torch
recommender = torch.load("models/baseline_recommender.pt")
model = recommender.model
```

---

## 5. Making Predictions Programmatically

Example: get top-10 recommendations for one user.

```python
import torch
import numpy as np

recommender = torch.load("models/baseline_recommender.pt")
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
python src/training/train_model.py
```

### Step 3 — Generate top-10 recommendations

```
python src/training/generate_top10.py
```

### Step 4 — View output

Open:

```
models/top10_recommendations_with_titles.csv
```

---
