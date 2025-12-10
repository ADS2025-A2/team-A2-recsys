import os
from typing import Any, Dict, Optional

import numpy as np
from spotlight.factorization.implicit import ImplicitFactorizationModel

# Fix OpenMP / MKL issues on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


class SpotlightBPRRecommender:
    """
    Thin wrapper around Spotlight's ImplicitFactorizationModel (BPR loss).

    This class is designed to have a simple, consistent interface so that
    the training script can instantiate it from a config and call:

        model = SpotlightBPRRecommender(**hyperparams)
        model.fit(train_interactions, verbose=True)

    Attributes
    ----------
    params : dict
        Hyperparameters used to construct the underlying Spotlight model.
    model : ImplicitFactorizationModel
        The underlying Spotlight model instance.
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
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        self.params: Dict[str, Any] = {
            "embedding_dim": embedding_dim,
            "n_iter": n_iter,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "l2": l2,
            "num_negative_samples": num_negative_samples,
            "use_cuda": use_cuda,
            "random_state": random_state,
        }

        self.model = ImplicitFactorizationModel(
            loss="bpr",
            embedding_dim=embedding_dim,
            n_iter=n_iter,
            batch_size=batch_size,
            learning_rate=learning_rate,
            l2=l2,
            num_negative_samples=num_negative_samples,
            use_cuda=use_cuda,
            random_state=random_state,
        )

    def fit(self, train_interactions, **fit_kwargs) -> "SpotlightBPRRecommender":
        """
        Fit the underlying Spotlight model.

        Parameters
        ----------
        train_interactions :
            Spotlight Interactions object used for training.
        **fit_kwargs :
            Extra keyword args forwarded to `ImplicitFactorizationModel.fit`,
            e.g. verbose=True.

        Returns
        -------
        self
        """
        self.model.fit(train_interactions, **fit_kwargs)
        return self

    def predict(self, user_id: int):
        """
        Predict scores for ALL items for a given user.

        Spotlight expects user_ids and an item range.
        """
        num_items = self.model._num_items
        item_ids = np.arange(num_items, dtype=np.int32)
        return self.model.predict(user_id, item_ids)

    def predict_for_user(self, user_id: int, item_ids=None):
        """
        Predict scores for a user, optionally restricted to a subset of items.
        """
        return self.model.predict(user_id, item_ids=item_ids)


if __name__ == "__main__":
    """
    Basic test:
    - Load a tiny sample of the dataset
    - Build train/val/test splits
    - Train the model for 1 epoch
    - Predict for a sample user
    """

    print("\n=== Running basic Spotlight recommender test ===")

    # Adjust this import to your package layout; assuming this file lives in src/
    try:
        from .load import prepare_datasets  # package-style import
    except ImportError:  # fallback for direct script execution
        from load import prepare_datasets

    # load a tiny sample for fast testing
    train, val, test = prepare_datasets(sample_size=10_000)

    # Initialize the model with tiny settings
    model = SpotlightBPRRecommender(
        embedding_dim=8,
        n_iter=1,          # just 1 epoch for the smoke test
        batch_size=512,
        learning_rate=1e-3,
        num_negative_samples=2,
        use_cuda=False,
    )

    # Fit the model
    print("\nTraining test model...")
    model.fit(train, verbose=True)

    # Take a user from the training set
    test_user_id = int(train.user_ids[0])
    print(f"\nPredicting for user {test_user_id}...")

    # Predict scores for all items
    scores = model.predict_for_user(test_user_id)

    print(f"Prediction vector length: {len(scores)}")
    print(f"Example scores: {scores[:10]}")

    print("\n=== Test completed successfully ===")
