# End-to-End Model Monitoring

This document describes the **end-to-end monitoring setup** for the recommendation system, covering data ingestion, model evaluation, metric tracking, automated checks, and retraining triggers. The goal is to ensure **model quality, stability, and business relevance over time**.

---

## 1. Monitoring Goals

The monitoring system is designed to:

* Continuously evaluate model performance on **real frontend user behavior**
* Detect **performance degradation** early
* Identify **data distribution shifts** (e.g. new users)
* Provide **traceability** of metrics over time
* Automatically trigger **model retraining** when necessary

---

## 2. High-Level Architecture

**End-to-end flow:**

1. Frontend users interact with the application
2. Ratings are stored in the application database
3. A scheduled evaluation job runs periodically
4. The trained model is evaluated on fresh frontend data
5. Metrics are logged to MLflow
6. Metrics are compared to thresholds and previous runs
7. Retraining is triggered if conditions are met

---

## 3. Data Monitoring (Input Layer)

### 3.1 Data Source

* Source: `user_data.db`
* Tables used:

  * `ratings`
  * `initial_ratings`

Only users who **completed the initial onboarding ratings** are included to ensure data quality.

```sql
SELECT r.username, r.movie, r.rating
FROM ratings r
JOIN initial_ratings i ON r.username = i.username
WHERE i.done = 1
```

### 3.2 Input Validation

The evaluation pipeline checks:

* Presence of frontend ratings
* Number of users
* Number of ratings
* Average ratings per user

If no valid data is found, the evaluation exits gracefully.

---

## 4. Model Evaluation Monitoring

### 4.1 Evaluation Schedule

* Triggered via **GitHub Actions**
* Runs:

  * Weekly (cron)
  * On-demand (manual trigger)

### 4.2 Metrics Computed

The following metrics are calculated using **Spotlight**:

| Metric                 | Description                            |
| ---------------------- | -------------------------------------- |
| `test_mrr`             | Mean Reciprocal Rank (ranking quality) |
| `precision_at_10`      | Precision@10                           |
| `recall_at_10`         | Recall@10                              |
| `num_users`            | Active frontend users                  |
| `num_ratings`          | Total ratings                          |
| `avg_ratings_per_user` | Engagement proxy                       |

These metrics capture both **model quality** and **data health**.

---

## 5. Metric Logging & Tracking (MLflow)

### 5.1 Experiment Setup

* Tracking backend: local `mlruns/`
* Experiment name: `evaluation_runs`

Each evaluation run logs:

* All computed metrics
* Timestamped run metadata

This enables:

* Historical comparisons
* Trend analysis
* Auditing of model behavior

---

## 6. Automated Monitoring Rules

### 6.1 Absolute Thresholds (Cold-Start Safety)

Minimum acceptable performance thresholds:

```python
THRESHOLDS = {
    "test_mrr": 0.02,
    "precision_at_10": 0.03,
    "recall_at_10": 0.03,
}
```

If any metric falls below its threshold → **retraining is triggered**.

---

### 6.2 Relative Performance Degradation

To detect silent regressions, the current run is compared to the **previous MLflow run**.

**MRR drop trigger:**

* If `test_mrr` drops by more than **10%** compared to the previous run

This protects against gradual model decay.

---

### 6.3 Data Drift Proxy: New User Growth

User base changes can invalidate an existing model.

Trigger condition:

* New users increase by more than **10%** compared to the previous evaluation

This acts as a **proxy for data distribution shift**.

---

## 7. Retraining Automation

If any trigger condition is met:

* A retraining flag is raised
* The training pipeline can be executed automatically

```python
if should_retrain(curr_metrics):
    train_pipeline()
```

This ensures the system remains adaptive as user behavior evolves.

---

## 8. CI/CD Integration

The evaluation pipeline is integrated into CI using **GitHub Actions**:

* Reproducible environment
* Dependency-controlled execution
* Scheduled and manual runs

This guarantees consistent monitoring independent of local machines.

---

## 9. Failure Handling & Safety

* Missing data → graceful exit
* No previous runs → skip comparisons
* Metrics always logged before decisions

The system prioritizes **robustness and explainability** over aggressive automation.

---

## 10. Benefits of This Monitoring Setup

* Early detection of model degradation
* Continuous visibility into model health
* Automated retraining triggers
* Clear separation of concerns (data, model, monitoring)
* Scalable foundation for production MLOps

---

## 11. Future Extensions

Possible enhancements:

* Alerting (Slack / Email)
* Model version comparison dashboards
* Feature-level data drift detection
* Shadow model evaluation
* Remote MLflow tracking backend

---

**This end-to-end monitoring setup ensures that the recommendation system remains accurate, reliable, and aligned with real user behavior over time.**


