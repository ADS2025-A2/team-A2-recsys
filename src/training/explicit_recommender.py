import os
import numpy as np

# Fix OpenMP / MKL issues on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from spotlight.factorization.explicit import ExplicitFactorizationModel


class SpotlightExplicitRegressionRecommender:
    def __init__(
        self,
        embedding_dim: int = 16,
        n_iter: int = 10,
        batch_size: int = 2048,
        learning_rate: float = 5e-3,
        l2: float = 1e-6,
        use_cuda: bool = False,
        random_state=None,
    ):
        """
        Wrapper around Spotlight's ExplicitFactorizationModel with
        regression loss.

        Interface mirrors SpotlightBPRRecommender:
        - .fit(train_interactions)
        - .predict(user_id)
        - .predict_for_user(user_id, item_ids=None)
        - .params dict for logging/MLflow
        """

        self.params = {
            "embedding_dim": embedding_dim,
            "n_iter": n_iter,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "l2": l2,
            "use_cuda": use_cuda,
            "random_state": random_state,
            "loss": "regression",
        }

        self.model = ExplicitFactorizationModel(
            loss="regression",
            embedding_dim=embedding_dim,
            n_iter=n_iter,
            batch_size=batch_size,
            learning_rate=learning_rate,
            l2=l2,
            use_cuda=use_cuda,
            random_state=random_state,
        )

    def fit(self, train_interactions):
        """Fit the explicit regression model on interactions."""
        self.model.fit(train_interactions, verbose=True)
        return self

    def predict(self, user_id):
        """
        Predict scores (expected ratings) for ALL items for a single user.
        Returns a numpy array of length = num_items.
        """
        num_items = self.model._num_items
        item_ids = np.arange(num_items)
        return self.model.predict(user_id, item_ids)

    def predict_for_user(self, user_id, item_ids=None):
        """
        Predict scores for a given user and optional list/array of item_ids.
        If item_ids is None, scores for all items are returned.
        """
        return self.model.predict(user_id, item_ids=item_ids)


if __name__ == "__main__":
    """
    Basic test:
    - Load a tiny sample of the dataset
    - Build train/val/test splits (from your `load.py`)
    - Train the explicit regression model for 1 epoch
    - Predict for a sample user
    """

    print("\n=== Running basic Spotlight explicit regression test ===")

    from load import prepare_datasets
    from spotlight.evaluation import mrr_score, precision_recall_score

    # Load a tiny sample for fast testing
    train, val, test = prepare_datasets(sample_size=10_000)

    # Initialize the model with small settings
    model = SpotlightExplicitRegressionRecommender(
        embedding_dim=8,
        n_iter=1,
        batch_size=512,
        learning_rate=1e-3,
        use_cuda=False,
        random_state=np.random.RandomState(42),
    )

    # Fit the model
    print("\nTraining test model...")
    model.fit(train)

    # Take a user from the training set
    test_user_id = int(train.user_ids[0])
    print(f"\nPredicting for user {test_user_id}...")

    # Predict scores for all items
    scores = model.predict_for_user(test_user_id)

    print(f"Prediction vector length: {len(scores)}")
    print(f"Example scores: {scores[:10]}")

    # Quick evaluation just to see it runs
    print("\nEvaluating with ranking metrics (MRR / P@10 / R@10)...")
    mrr = mrr_score(model.model, test, train=train).mean()
    precision, recall = precision_recall_score(model.model, test, train=train, k=10)
    print(f"MRR:            {mrr:.4f}")
    print(f"Precision@10:   {precision.mean():.4f}")
    print(f"Recall@10:      {recall.mean():.4f}")

    print("\n=== Test completed successfully ===")
