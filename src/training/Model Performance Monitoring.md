# Model Performance Monitoring & Retraining Policy #

This document defines the **key metrics**, **evaluation procedures**, and **retraining triggers** for the Spotlight BPR Movie Recommendation Model.

The goal is to ensure that recommendations stay relevant as user-level interaction data evolves (e.g., new ratings, updated watchlists, changes in trends).

---

## **1. Key Metrics**

We monitor **ranking-based recommendation metrics**, computed during the training pipeline:

### **1.1. Mean Reciprocal Rank (MRR)**

**Definition:**
MRR measures how highly the model ranks the first relevant item.
**Interpretation:** Higher = better. Sensitive to early-ranking errors.

### **1.2. Precision@K**

**Definition:**
Percentage of the top-K recommendations that are relevant.
**Interpretation:** Measures recommendation *quality*.

### **1.3. Recall@K**

**Definition:**
Percentage of relevant items that appear in the top-K recommendations.
**Interpretation:** Measures *coverage* of the user’s interests.

### **1.4. Offline Monitoring Frequency**

Metrics are logged on every training run in MLflow:

| Step            | Metric                                        |
| --------------- | --------------------------------------------- |
| Validation step | `val_mrr`                                     |
| Test step       | `test_mrr`, `precision_at_10`, `recall_at_10` |
| Run metadata    | hyperparameters, dataset sizes                |

---

# **2. Thresholds for Model Retraining**

The model should be **automatically retrained** when one or more of the following degradation criteria is met:

---

## **2.1. Absolute Performance Thresholds**

These are minimum acceptable metric values based on baseline performance from the initial model training.

| Metric           | Minimum Threshold | Rationale                                              |
| ---------------- | ----------------- | ------------------------------------------------------ |
| **Test MRR**     | **≥ 0.075**       | MRR is sensitive to ranking errors — ensures relevance |
| **Precision@10** | **≥ 0.085**       | Avoid recommending irrelevant movies                   |
| **Recall@10**    | **≥ 0.050**       | Ensure coverage of user interests                      |

If any single metric falls **below threshold for 2 consecutive runs**, retraining is triggered.

---

## **2.2. Relative Degradation Thresholds**

Retraining is triggered when a metric drops significantly compared to a stable baseline:

| Condition                                                           | Trigger |
| ------------------------------------------------------------------- | ------- |
| **MRR decreases by more than 10%** from the previous stable version | Retrain |
| **Precision@10 or Recall@10 decreases by more than 15%**            | Retrain |

This protects against gradual model drift.

---

## **2.3. User Data Drift Triggers**

| User Drift Event                                                     | Trigger |
| -------------------------------------------------------------------- | ------- |
| **+20% new ratings** since last retraining                           | retrain |
| **+10% new users** added                                             | retrain |
| **>15% changes in watchlist patterns** (e.g., trending genres shift) | retrain |
| **Data quality anomaly** (missing/invalid rating spikes)             | retrain |

---

# **3. Runtime Trigger for a Single-User Recommendation Refresh**

When a user updates:

* new ratings
* updated watchlist
* changed initial preferences

We **do not retrain the full model**.

Instead, we:

1. Embed the user using the existing Spotlight model
   – call `model.predict(user_id)` for updated scores
2. Re-generate top-10 recommendations
3. Store them in the user’s recommendation table

---

# **4. Full Retraining Workflow (CI/CD)**

Retraining is automatically triggered when:

✔ Metric drop > threshold
✔ Data drift exceeds configured limits
✔ Scheduled retraining interval reached (e.g., every 14 days)

### **Pipeline Behavior**

* Runs `train_model.py`
* Logs metrics to MLflow
* Registers new model version if thresholds are met
* Tags repo with new version (`git tag 1.X`)
* Generates new top-10 recommendations for all users
* Exports them to:

```
models/top10_recommendations_with_titles.csv
```


