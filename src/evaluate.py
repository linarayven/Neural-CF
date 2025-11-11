import torch
from src.dataset import get_dataloader
from src.model import NeuralCF
from sklearn.metrics import mean_squared_error
import numpy as np

# Пути
RATINGS_PATH = 'data/ratings_ascii.dat'
MOVIES_PATH = 'data/movies_ascii.dat'

# Данные
dataloader, num_users, num_movies, _, _ = get_dataloader(RATINGS_PATH, MOVIES_PATH, batch_size=64)

# Модель
model = NeuralCF(num_users, num_movies)
model.load_state_dict(torch.load('neural_cf_model.pth'))
model.eval()

# Сбор предсказаний и истинных значений
y_true = []
y_pred = []

with torch.no_grad():
    for user, movie, rating in dataloader:
        output = model(user, movie)
        y_true.extend(rating.numpy())
        y_pred.extend(output.numpy())

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: {rmse:.4f}")
