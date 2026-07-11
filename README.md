# Neural Collaborative Filtering (Neural-CF)

**A deep learning-based movie recommendation engine** built with PyTorch, developed as a practical implementation of Neural Collaborative Filtering (NCF) for the *Artificial Neural Networks* course. This project demonstrates how neural networks can learn complex user-item interaction patterns for personalized recommendations.

---

## Overview

Neural-CF explores how neural networks can outperform traditional matrix factorization on collaborative filtering tasks. Rather than relying on hand-crafted similarity metrics, the model learns non-linear user-item relationships through embeddings and multi-layer perceptrons.

The project covers the full ML pipeline: data preprocessing and categorical encoding, model architecture design, training with MSE loss, evaluation with RMSE metric, and an optimized inference layer with optional caching for fast recommendation generation.

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

1. **Embedding Layer** — maps user and item IDs to dense latent vectors (32-dimensional by default)
2. **Concatenation** — merges user and item embeddings into a joint representation (64-dimensional input)
3. **Dense (MLP) Layers** — multi-layer perceptron with ReLU activation and dropout (default: [64, 32] neurons)
4. **Output Layer** — single neuron with sigmoid activation scaled to **[1, 5] range** for rating prediction

Trained with **Adam** optimizer and **MSE (Mean Squared Error) loss**. Final model weights are saved as `neural_cf_model.pth` with metadata after training completes.

---

## Dataset

Built on the **MovieLens 100K** dataset:

| File | Size | Description |
|---|---|---|
| `ratings_ascii.dat` | ~25 MB | `userId::movieId::rating::timestamp` (100K+ ratings) |
| `movies_ascii.dat` | ~175 KB | `movieId::title::genres` (~1,600 movies) |
| `users_ascii.dat` | ~140 KB | `userId::gender::age::occupation::zip` (~600–1,000 users) |

The interaction matrix is highly sparse, which is representative of real-world recommendation scenarios. User and movie IDs are encoded using `LabelEncoder` for efficient neural network processing.

---

## Project Structure

```
Neural-CF/
├── src/
│   ├── dataset.py         # Data loading, preprocessing, and LabelEncoder for categorical IDs
│   ├── model.py            # Neural CF model architecture
│   ├── train.py             # Training loop with MSE loss and Adam optimizer
│   ├── utils.py              # Recommendation generation, stats, formatting
│   └── evaluate.py            # Model evaluation (RMSE metric)
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

## Model Training

The training pipeline:
- Loads MovieLens dataset and encodes categorical user/movie IDs using `LabelEncoder`
- Trains for 15 epochs with batch size 128
- Uses Adam optimizer (learning rate: 0.001, weight decay: 1e-5) with gradient clipping
- Monitors training loss and sample predictions each epoch
- Saves final model as `neural_cf_model.pth` with metadata (num_users, num_movies)

Evaluation uses **RMSE (Root Mean Squared Error)** to measure prediction accuracy on held-out data.

---

## Recommendation Engine

### Standard Mode
Generates top-K recommendations by:
1. Computing predictions for all unrated movies in batches (5000 movies per batch)
2. Ranking by model confidence scores
3. Retrieving movie metadata and real average ratings from dataset

### Fast Mode
Uses `@lru_cache` decorator to cache user data and movie ratings, reducing repeated file I/O operations (~2× faster).

### Ultra-fast Mode
Optimized batch processing for large candidate sets, reducing memory footprint and computation time (~5× faster).

### Filtering Options
- **Genre**: Filter recommendations by movie genre (Action, Comedy, Drama, etc.)
- **Age**: Apply content-appropriate filters:
  - Age ≤ 13: Recommend only children's movies
  - Age ≤ 18: Exclude horror, thriller, crime, and film-noir
- **Gender**: Restrict recommendations to users matching specified gender

---

## About the Project

Neural-CF was developed as a practical implementation for the *Artificial Neural Networks* course, applying deep learning to collaborative filtering and recommendation systems — from data preprocessing through model deployment and user-facing recommendations.

---

## Acknowledgments

- **MovieLens** for the public recommendation dataset
- **PyTorch** for the deep learning framework
