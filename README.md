# Neural Collaborative Filtering (Neural-CF)

**A deep learning-based movie recommendation engine** built with PyTorch, developed as a practical implementation of Neural Collaborative Filtering (NCF) for the *Artificial Neural Networks* course. The system learns latent user and item representations to predict ratings and generate personalized movie recommendations from real-world interaction data.

---

## Overview

Neural-CF explores how neural networks can outperform traditional matrix factorization on collaborative filtering tasks. Rather than relying on hand-crafted similarity metrics, the model learns non-linear user–item interaction patterns end-to-end from the MovieLens dataset, then exposes them through an interactive CLI for searching users, generating recommendations, and inspecting dataset statistics.

The project covers the full ML pipeline: data preprocessing and encoding, model architecture design, training, evaluation (Hit Rate, NDCG), and a performance-optimized inference layer with caching for fast repeated queries.

---

## Key Features

- **Neural network-based collaborative filtering** — embedding layers + MLP learn latent user/item representations
- **User search by demographics** — filter by age, gender, and occupation
- **Personalized recommendations** — top-K movie predictions per user, with optional genre/age/gender filters
- **Three performance modes** — standard, cached ("fast", ~2×), and fully optimized ("ultra-fast", ~5×) inference
- **Dataset statistics** — rating distributions, sparsity metrics, demographic breakdowns
- **CSV export** — save generated recommendations for further analysis
- **Bilingual CLI** — interactive menu available in Ukrainian

---

## Technology Stack

| Layer | Technologies |
|---|---|
| **Language** | Python 3.x |
| **Deep Learning** | PyTorch |
| **Data Processing** | Pandas, NumPy, Scikit-learn |

---

## Model Architecture

The NCF model combines learned embeddings with a multi-layer perceptron to capture non-linear user–item interactions:

1. **Embedding Layer** — maps user and item IDs to dense latent vectors
2. **Concatenation** — merges user and item embeddings into a joint representation
3. **Dense (MLP) Layers** — learn non-linear interaction patterns between users and items
4. **Output Layer** — sigmoid activation producing a predicted rating (0–5 scale)

Trained with **Adam** optimizer and **MSE loss**, with model checkpointing during training.

---

## Dataset

Built on the **MovieLens 100K** dataset:

| File | Size | Description |
|---|---|---|
| `ratings_ascii.dat` | ~25 MB | `userId::movieId::rating::timestamp` (100K+ ratings) |
| `movies_ascii.dat` | ~175 KB | `movieId::title::genres` (~1,600 movies) |
| `users_ascii.dat` | ~140 KB | `userId::gender::age::occupation::zip` (~600–1,000 users) |

The interaction matrix is highly sparse, which is representative of real-world recommendation scenarios.

---

## Project Structure

```
Neural-CF/
├── src/
│   ├── dataset.py         # Data loading and preprocessing
│   ├── model.py            # Neural CF model architecture
│   ├── train.py             # Training loop and checkpointing
│   ├── utils.py              # Recommendation generation, stats, formatting
│   └── evaluate.py            # Evaluation metrics (HR@K, NDCG)
├── data/                        # MovieLens dataset files
├── filesArchive/                 # Archived earlier iterations
├── neural_cf_model.pth            # Pre-trained model weights
├── test_recommendations.py         # Interactive CLI entry point
└── README.md
```

---

## Getting Started

**Prerequisites**

```bash
pip install torch pandas numpy scikit-learn
```

**Setup**

```bash
git clone https://github.com/linarayven/Neural-CF.git
cd Neural-CF
python test_recommendations.py
```

The pre-trained model and dataset are included, so the interactive system runs out of the box — no training required to explore recommendations.

**Interactive menu:**

1. Find users by criteria (age / gender / occupation)
2. Get recommendations for a specific user ID
3. Quick test with a sample user
4. View dataset statistics
5–6. Fast / ultra-fast recommendation modes (cached inference)
7. Clear cache
8. Exit

To point the CLI at custom data or model paths, update the corresponding constants in `test_recommendations.py`.

---

## Evaluation

Model quality is assessed using standard top-K recommendation metrics:
- **Hit Rate (HR@K)** — whether a relevant item appears in the top-K recommendations
- **NDCG (Normalized Discounted Cumulative Gain)** — rewards relevant items ranked higher in the list

---

## About the Project

Neural-CF was developed as a practical implementation for the *Artificial Neural Networks* course, applying deep learning to collaborative filtering and recommendation systems — from data preprocessing and model design through training, evaluation, and a performance-optimized inference layer.

## Acknowledgments

- **MovieLens** for the public recommendation dataset
- **PyTorch** for the deep learning framework
