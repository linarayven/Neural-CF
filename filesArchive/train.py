import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Добавляем путь к src в PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dataset import get_dataloader
from model import NeuralCF
import time

# Пути к файлам
RATINGS_PATH = 'data/ratings_ascii.dat'
MOVIES_PATH = 'data/movies_ascii.dat'
USERS_PATH = 'data/users_ascii.dat'

# Параметры
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001

print("📊 Загрузка данных...")
dataloader, num_users, num_movies, _, _, dataset = get_dataloader(RATINGS_PATH, MOVIES_PATH, USERS_PATH, BATCH_SIZE)

print(f"📈 Статистика данных:")
print(f"   👥 Пользователей: {num_users}")
print(f"   🎬 Фильмов: {num_movies}")
print(f"   ⭐ Оценок: {len(dataset.ratings)}")
print(f"   📊 Средний рейтинг: {dataset.ratings['rating'].mean():.2f}")

# Создаем модель
model = NeuralCF(num_users, num_movies)

# Оптимизатор
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
criterion = nn.MSELoss()

print("\n🎯 Начало обучения...")
print(f"   Модель: {sum(p.numel() for p in model.parameters()):,} параметров")

# Обучение
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, (user, movie, rating) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(user, movie)
        loss = criterion(output, rating)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(dataloader)

    # Проверяем предсказания
    with torch.no_grad():
        test_users = torch.tensor([0, 1, 2], dtype=torch.long)
        test_movies = torch.tensor([0, 1, 2], dtype=torch.long)
        test_preds = model(test_users, test_movies)
        print(f"📊 Примеры предсказаний: {[f'{p:.2f}' for p in test_preds.tolist()]}")

    print(f"✅ Epoch {epoch + 1}/{EPOCHS} завершена")
    print(f"   ⏱️  Время: {epoch_time:.1f}с")
    print(f"   📉 Loss: {avg_loss:.4f}")
    print("-" * 50)

# Сохраняем модель
torch.save({
    'model_state_dict': model.state_dict(),
    'num_users': num_users,
    'num_movies': num_movies,
}, 'neural_cf_model.pth')

print("💾 Модель сохранена!")