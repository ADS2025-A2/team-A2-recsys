import os
import numpy as np

# Fix macOS MKL/OMP issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from spotlight.factorization.implicit import ImplicitFactorizationModel


class SpotlightAdaptiveRecommender:
    """
    Implicit matrix factorization using Spotlight's adaptive hinge loss.
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        n_iter: int = 10,
        batch_size: int = 2048,
        learning_rate: float = 5e-3,
        l2: float = 1e-6,
        num_negative_samples: int = 3,
        use_cuda: bool = False,
        random_state=None,
    ):
        # Store hyperparameters for MLflow logging
        self.params = {
            "model_type": "adaptive_hinge_mf",
            "embedding_dim": embedding_dim,
            "n_iter": n_iter,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "l2": l2,
            "num_negative_samples": num_negative_samples,
            "use_cuda": use_cuda,
            "random_state": random_state,
        }

        # Spotlight model using adaptive hinge loss
        self.model = ImplicitFactorizationModel(
            loss="adaptive_hinge",
            embedding_dim=embedding_dim,
            n_iter=n_iter,
            batch_size=batch_size,
            learning_rate=learning_rate,
            l2=l2,
            num_negative_samples=num_negative_samples,
            use_cuda=use_cuda,
            random_state=random_state,
        )

    def fit(self, train_interactions):
        """Train the model."""
        self.model.fit(train_interactions, verbose=True)
        return self

    def predict(self, user_id):
        """Predict scores for ALL items for a single user."""
        num_items = self.model._num_items
        item_ids = np.arange(num_items)
        return self.model.predict(user_id, item_ids)

    def predict_for_user(self, user_id, item_ids=None):
        """Predict for a specific set of item IDs (or all if None)."""
        return self.model.predict(user_id, item_ids=item_ids)


if __name__ == "__main__":
    # Small functional test
    from load import prepare_datasets

    print("\n=== Running Spotlight Adaptive MF test ===")

    train, val, test = prepare_datasets(sample_size=10_000)

    model = SpotlightAdaptiveRecommender(
        embedding_dim=8,
        n_iter=1,
        batch_size=512,
        learning_rate=1e-3,
        num_negative_samples=2,
        use_cuda=False,
    )

    print("\nTraining test model...")
    model.fit(train)

    test_user_id = train.user_ids[0]
    print(f"\nPredicting for user {test_user_id}...")
    scores = model.predict_for_user(test_user_id)

    print(f"Prediction vector length: {len(scores)}")
    print(f"Example scores: {scores[:10]}")
    print("\n=== Adaptive MF test completed successfully ===")
