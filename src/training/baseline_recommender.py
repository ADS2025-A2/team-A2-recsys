
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
from spotlight.factorization.implicit import ImplicitFactorizationModel

class SpotlightBPRRecommender:
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
        self.params = {
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

    def fit(self, train_interactions):
        self.model.fit(train_interactions, verbose=True)
        return self
    
    def predict(self, user_id):
        # Spotlight expects user_ids AND an item range
        # Predict scores for all items for the given user
        num_items = self.model._num_items
        item_ids = np.arange(num_items)

        return self.model.predict(user_id, item_ids)

    def predict_for_user(self, user_id, item_ids=None):
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

    from load import prepare_datasets

    # load a tiny sample for fast testing
    train, val, test = prepare_datasets(sample_size=10_000)

    # Initialize the model with tiny settings
    model = SpotlightBPRRecommender(
        embedding_dim=8,
        n_iter=1,          
        batch_size=512,
        learning_rate=1e-3,
        num_negative_samples=2,
        use_cuda=False,
    )

    # Fit the model
    print("\nTraining test model...")
    model.fit(train)

    # Take a user from the training set
    test_user_id = train.user_ids[0]
    print(f"\nPredicting for user {test_user_id}...")

    # Predict scores for all items
    scores = model.predict_for_user(test_user_id)

    print(f"Prediction vector length: {len(scores)}")
    print(f"Example scores: {scores[:10]}")

    print("\n=== Test completed successfully ===")
