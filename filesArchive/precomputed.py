# precomputed.py
import pickle
import pandas as pd
import numpy as np
from functools import lru_cache
import os
import time


class PrecomputedData:
    def __init__(self, dataset):
        self.dataset = dataset
        self._cache_file = "precomputed_cache.pkl"
        self.load_or_compute()

    def load_or_compute(self):
        """Загружает из файла или вычисляет заново"""
        try:
            if os.path.exists(self._cache_file):
                print("🔄 Завантаження попередніх обчислень з кешу...")
                with open(self._cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Восстанавливаем только вычисленные данные
                    for key, value in cached_data.items():
                        setattr(self, key, value)
                print("✅ Попередні обчислення успішно завантажено")
                return
        except Exception as e:
            print(f"⚠️ Не вдалося завантажити кеш: {e}")

        print("🔄 Обчислення попередніх даних...")
        self._compute_all()
        self._save_to_cache()

    def _compute_all(self):
        """Вычисляет все необходимые данные один раз"""
        start_time = time.time()

        print("📊 Обчислення середніх рейтингів...")
        # 1. Средние рейтинги фильмов
        self.avg_ratings = self.dataset.ratings.groupby('movieId')['rating'].mean().to_dict()

        print("👥 Створення матриці користувач-фільм...")
        # 2. Матрица "пользователь-фильм" для быстрого поиска
        self.user_rated_movies = self.dataset.ratings.groupby('userId')['movieId'].apply(set).to_dict()

        print("🎬 Отримання унікальних фільмів...")
        # 3. Все уникальные фильмы
        self.all_movies = np.unique(self.dataset.ratings['movieId'].values)
        self.all_movies = self.all_movies[~np.isnan(self.all_movies)].astype(int)

        print("🎭 Індексація жанрів...")
        # 4. Жанры для быстрой фильтрации
        if hasattr(self.dataset, 'movies') and self.dataset.movies is not None:
            self.movie_genres = self.dataset.movies.set_index('movieId')['genres'].to_dict()
            self.movie_titles = self.dataset.movies.set_index('movieId')['title'].to_dict()
        else:
            self.movie_genres = {}
            self.movie_titles = {}

        computation_time = time.time() - start_time
        print(f"✅ Попередні обчислення завершено за {computation_time:.2f} сек")

    def _save_to_cache(self):
        """Сохраняет в файл для будущих запусков"""
        try:
            cache_data = {
                'avg_ratings': self.avg_ratings,
                'user_rated_movies': self.user_rated_movies,
                'all_movies': self.all_movies,
                'movie_genres': self.movie_genres,
                'movie_titles': self.movie_titles
            }

            with open(self._cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"💾 Попередні обчислення збережено в {self._cache_file}")
        except Exception as e:
            print(f"⚠️ Не вдалося зберегти кеш: {e}")

    def get_unrated_movies_fast(self, user_id):
        """Быстро получает непросмотренные фильмы"""
        rated_movies = self.user_rated_movies.get(user_id, set())
        unrated_movies = np.array([m for m in self.all_movies if m not in rated_movies])
        return unrated_movies

    def get_movie_info_fast(self, movie_id):
        """Быстро получает информацию о фильме"""
        return {
            'genres': self.movie_genres.get(movie_id, ''),
            'title': self.movie_titles.get(movie_id, 'Unknown'),
            'avg_rating': self.avg_ratings.get(movie_id, 0.0)
        }


# Глобальный экземпляр
precomputed = None


def init_precomputed(dataset):
    global precomputed
    precomputed = PrecomputedData(dataset)
    return precomputed


def clear_cache():
    """Очищает кеш (использовать при изменении данных)"""
    global precomputed
    precomputed = None
    try:
        if os.path.exists("precomputed_cache.pkl"):
            os.remove("precomputed_cache.pkl")
            print("✅ Кеш очищено")
    except:
        pass