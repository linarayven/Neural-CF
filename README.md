# Neural Collaborative Filtering (Neural-CF)

A neural network-based collaborative filtering system for movie recommendations, developed as part of the "Artificial Neural Networks" course. This project implements a Neural Collaborative Filtering (NCF) model to predict user preferences and generate personalized movie recommendations.

## 📋 Overview

Neural-CF is an interactive movie recommendation system that leverages deep learning to understand user-movie interactions. The system is trained on the MovieLens dataset and provides an intuitive interface for searching users by demographic criteria and generating personalized recommendations.

**Key Features:**
- 🎯 Neural network-based collaborative filtering
- 👥 User search by age, gender, and occupation
- 🎬 Personalized movie recommendations
- ⚡ Optimized recommendation engine with caching
- 📊 Comprehensive user and data statistics
- 💾 Export recommendations to CSV

## 🏗️ Project Structure

```
Neural-CF/
├── src/                          # Source code modules
│   ├── __init__.py              # Package initialization
│   ├── dataset.py               # Data loading and preprocessing
│   ├── model.py                 # Neural CF model architecture
│   ├── train.py                 # Model training script
│   ├── utils.py                 # Utility functions and recommendations
│   └── evaluate.py              # Model evaluation metrics
├── data/                         # Dataset directory
│   ├── ratings_ascii.dat        # User-movie ratings (100K+ ratings)
│   ├── movies_ascii.dat         # Movie metadata
│   └── users_ascii.dat          # User demographic information
├── filesArchive/                # Archived versions of scripts
├── neural_cf_model.pth          # Pre-trained model weights
├── test_recommendations.py       # Interactive recommendation interface
└── README.md                     # This file
```

## 🛠️ Technology Stack

- **Python 3.x** - Primary programming language (97.7% of codebase)
- **PyTorch** - Deep learning framework for the neural network
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Data encoding and preprocessing

## 📊 Dataset

The project uses the **MovieLens 100K Dataset**:
- **Users**: ~600-1000 users with demographic data (age, gender, occupation)
- **Movies**: ~1600+ movies with genres
- **Ratings**: 100,000+ ratings (sparse matrix)
- **Sparsity**: High sparsity (most user-movie pairs are unrated)

### Data Files

| File | Size | Description |
|------|------|-------------|
| `ratings_ascii.dat` | ~25 MB | User ratings (format: userId::movieId::rating::timestamp) |
| `movies_ascii.dat` | ~175 KB | Movie information (format: movieId::title::genres) |
| `users_ascii.dat` | ~140 KB | User demographics (format: userId::gender::age::occupation::zip) |

## 🤖 Model Architecture

The Neural Collaborative Filtering model consists of:

- **Embedding Layer**: Learns latent representations for users and items
- **Concatenation**: Combines user and item embeddings
- **Dense Layers**: Multi-layer perceptron (MLP) for non-linear feature interaction
- **Output Layer**: Sigmoid activation for rating prediction (0-5 scale)

The model learns to capture complex user-item interactions better than traditional matrix factorization.

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch pandas numpy scikit-learn
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/linarayven/Neural-CF.git
cd Neural-CF
```

2. Ensure data files are in the `data/` directory

3. The pre-trained model (`neural_cf_model.pth`) is included

### Running the Interactive System

```bash
python test_recommendations.py
```

This launches an interactive menu with the following options:

1. **🔍 Find users by criteria** - Search users by age, gender, or occupation
2. **🎯 Test specific user** - Get recommendations for a user ID
3. **🚀 Quick test** - Rapid test with user 5
4. **📊 Data statistics** - View dataset statistics
5. **⚡ Fast test** - Optimized recommendations with caching
6. **🚀 Ultra-fast test** - Maximum performance mode
7. **🗑️ Clear cache** - Reset precomputed data
8. **❌ Exit** - Quit the application

### Example Usage

```bash
$ python test_recommendations.py

🎬 ІНТЕРАКТИВНА СИСТЕМА РЕКОМЕНДАЦІЙ ФІЛЬМІВ (ОПТИМІЗОВАНА)
==================================================

📋 Головне меню:
1. 🔍 Знайти користувачів за критеріями
2. 🎯 Тестувати конкретного користувача (за ID)
3. 🚀 ШВИДКИЙ тест (користувач 5)
4. 📊 Статистика даних
5. ⚡ Надзвичайно швидкий тест
6. 🚀 УЛЬТРА-швидкий тест
7. 🗑️  Очистити кеш
8. ❌ Вийти

🎯 Оберіть опцію (1-8): 1

=== ПОШУК КОРИСТУВАЧІВ ЗА КРИТЕРІЯМИ ===
💡 Залишіть поле порожнім, щоб пропустити критерій

🔢 Вік: 25
👥 Стать (M/F): M
💼 Професія (0-20): 
```

## 📁 Source Modules

### `src/dataset.py`
Handles data loading and preprocessing:
- Loads ratings, movies, and user data
- Creates data loaders for training
- Encodes categorical variables

### `src/model.py`
Defines the Neural CF architecture:
- Embedding layers for users and items
- MLP for feature interaction
- Output layer for rating prediction

### `src/train.py`
Training script:
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Model checkpoint saving

### `src/utils.py`
Comprehensive utility functions:
- `get_recommendations()` - Generate top-K recommendations
- `get_recommendations_filtered()` - Recommendations with genre/age/gender filters
- `get_recommendations_fast()` - Optimized with caching
- `get_recommendations_ultra_fast()` - Maximum performance mode
- `get_user_stats()` - User interaction statistics
- `display_recommendations_with_scores()` - Formatted output

### `src/evaluate.py`
Model evaluation metrics:
- Hit Rate (HR@K)
- Normalized Discounted Cumulative Gain (NDCG)

## 🎯 Key Features

### User Search
Search for users by demographics:
- **Age**: 1, 18, 25, 35, 45, 50, 56 (standard values)
- **Gender**: M (Male), F (Female)
- **Occupation**: 0-20 (various job categories)

### Recommendation Filters
Get tailored recommendations:
- **Genre filtering**: Action, Comedy, Drama, etc.
- **Age-based**: Recommend movies for specific age groups
- **Gender-based**: Filter by user gender

### Performance Modes
Three recommendation engines:
1. **Standard** - Complete analysis
2. **Fast** - Precomputed caching (~2x speedup)
3. **Ultra-fast** - Maximum optimization (~5x speedup)

## 📈 Statistics & Insights

The system provides:
- Total users and movies count
- Rating distribution and averages
- Data sparsity metrics
- Age and gender distribution
- User rating statistics

## 💾 Output

Recommendations can be exported as CSV files containing:
- Movie ID and Title
- Predicted Rating Score
- Genres

Format: `recommendations_user_[ID].csv`

## 📝 Language Support

The interface supports **Ukrainian** language for user prompts and messages, with emojis for better visualization.

## 🔧 Configuration

To modify data paths, edit the paths in `test_recommendations.py`:

```python
self.RATINGS_PATH = r"path/to/ratings_ascii.dat"
self.MOVIES_PATH = r"path/to/movies_ascii.dat"
self.USERS_PATH = r"path/to/users_ascii.dat"
self.MODEL_PATH = r"path/to/neural_cf_model.pth"
```

## 📚 Course Information

This project was developed for the "Artificial Neural Networks" course as a practical application of:
- Collaborative Filtering techniques
- Deep Learning with PyTorch
- Recommendation Systems
- Data preprocessing and feature engineering

## 📁 Archived Files

The `filesArchive/` directory contains previous versions and iterations of the recommendation system for reference.

## 🤝 Contributing

This is a course project. Suggestions and improvements are welcome!

## 📄 License

This project is a course assignment and is available for educational purposes.

## 📧 Author

**linarayven**

## 🙏 Acknowledgments

- MovieLens dataset for the public recommendation data
- PyTorch team for the deep learning framework
- Course instructors for guidance and feedback

---

**Note**: This is a backup working version with all necessary data and model weights included for easy setup and testing.
