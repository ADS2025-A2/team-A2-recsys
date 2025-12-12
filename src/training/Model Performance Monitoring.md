# Model Performance Monitoring & Retraining Policy #

This document defines the **key metrics**, **evaluation procedures**, and **retraining triggers** for the Spotlight BPR Movie Recommendation Model.

The goal is to ensure that recommendations stay relevant as user-level interaction data evolves (e.g., new ratings, updated watchlists, changes in trends).

---

## **1. Key Metrics**

We monitor **ranking-based recommendation metrics**, computed on frontend users who have completed initial ratings.

### **1.1. Mean Reciprocal Rank (MRR)**

**Definition:**
MRR measures the rank of the first relevant item in the recommended list for a user.
**Interpretation:** Higher MRR indicates that relevant items appear early in the recommendation list. This metric is robust even for users with few ratings.
Current threshold: 0.02

### **1.2. Precision@K**

**Definition:**
Percentage of the top-10 recommended items that are relevant to the user.
**Interpretation:** 
Reflects recommendation accuracy. For frontend users with few ratings, a lower threshold is expected due to the cold-start problem.
Current threshold for frontend users: 0.03 (can be increased as users provide more ratings).

### **1.3. Recall@K**

**Definition:**
Percentage of relevant items that appear in the top-K recommendations.
**Interpretation:** 
Reflects coverage of the user’s preferences.
Current threshold for frontend users: 0.03 (can be increased as users provide more ratings).

### **1.4. Offline Monitoring Frequency**

Metrics are logged on every training run in MLflow:

| Step            | Metric                                        |
| --------------- | --------------------------------------------- |
| Test step       | `test_mrr`, `precision_10`, `recall_10` |
| Run metadata    | hyperparameters, dataset sizes                |

---

# **2. Thresholds for Model Retraining**

Retraining is required to ensure that the recommendation model remains accurate and relevant, especially as new users and interactions are added. The triggers are based on user activity, model performance, and drift detection.

### **2.1. User Growth-Based Triggers

**New users threshold:**
* Trigger retraining when the number of new frontend users since the last training run exceeds a defined percentage of total users (e.g., 10–20%).

**Ratings accumulation:**
* Retrain when the average number of ratings per user increases significantly, e.g., when the new batch of users have submitted ≥5–10 ratings each.

### **2.2. Metric-Based Triggers

**Cold-start metrics decline:**
* Trigger retraining if MRR, Precision@10, or Recall@10 for new users fall below predefined thresholds for consecutive evaluation periods.
Example: Precision@10 < 0.03 or Recall@10 < 0.03 for 3 evaluation runs in a row.

**Overall performance drift:**
* Track metrics across all frontend users.
Significant decline (e.g., >10% drop in MRR) triggers retraining.
Detects trends such as changes in user behavior or new popular movies.

### **2.3. Frequency-Based Triggers
* Schedule regular retraining (biweekly) even if thresholds are not crossed, to incorporate accumulated feedback.

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


