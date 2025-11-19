# Chosen Recommender Model

## 1. Library

We use the **Spotlight** library by Maciej Kula, a PyTorch-based toolkit for deep and shallow recommender models. It provides ready-made implementations for factorization and sequential recommenders, plus utilities for datasets and evaluation. :contentReference[oaicite:0]{index=0}

## 2. Model Type

We use **ImplicitFactorizationModel** from `spotlight.factorization.implicit` with a **BPR (Bayesian Personalized Ranking) loss**. :contentReference[oaicite:1]{index=1}

### Why implicit MF?

- Our data is best viewed as **implicit feedback** (interactions such as clicks/plays/views) rather than “all ratings for all items”.
- Implicit MF learns **latent vectors for users and items**; their dot product is the relevance score. This follows the matrix factorization approach of Koren et al. (2009). :contentReference[oaicite:2]{index=2}
- Spotlight’s implementation handles:
  - Negative sampling for unobserved user–item pairs.
  - GPU training via PyTorch.
  - Built-in evaluation utilities.

### Why BPR loss?

BPR is a **pairwise ranking loss**: for each user we train the model to rank interacted items higher than randomly sampled non-interacted items. This directly optimizes ranking quality, which is usually what we care about in a recommender UI. :contentReference[oaicite:3]{index=3}

### High-level formulation

- Each user \(u\) and item \(i\) get latent vectors \(p_u, q_i \in \mathbb{R}^k\).
- The relevance score is \( \hat{r}_{ui} = p_u^\top q_i \).
- For an observed positive pair \((u, i)\) and a sampled negative item \(j\), BPR maximizes:
  \[
  \ln \sigma(\hat{r}_{ui} - \hat{r}_{uj}) - \lambda\|\Theta\|^2
  \]
  where \(\sigma\) is the sigmoid and \(\Theta\) are all parameters.

### Evaluation

We evaluate with Spotlight’s built-in metrics: :contentReference[oaicite:4]{index=4}

- **MRR** (`mrr_score`): mean reciprocal rank per user.
- **Precision@k / Recall@k** (`precision_recall_score`): ranking quality for top-k recommendations.

These metrics operate directly on the ranking induced by the model over all items.

### Hyperparameters (initial defaults)

- `loss = "bpr"`
- `embedding_dim = 64`
- `n_iter = 20`
- `batch_size = 1024`
- `learning_rate = 1e-3`
- `l2 = 1e-6`
- `num_negative_samples = 5` (for adaptive hinge; unused for standard BPR but harmless)
- `use_cuda = True` if a GPU is available

These values are reasonable MovieLens-scale defaults and can be tuned later.