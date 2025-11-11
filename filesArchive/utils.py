# utils.py
import torch
import pandas as pd
import numpy as np
import os
from functools import lru_cache
from typing import Dict, List, Optional, Union
import time

# Импортируем наш модуль предвычислений
from .precomputed import precomputed


@lru_cache(maxsize=128)
def _load_users_data(users_path: str) -> pd.DataFrame:
    """Кешування даних користувачів"""
    try:
        return pd.read_csv(
            users_path,
            sep='::',
            engine='python',
            names=['userId', 'gender', 'age', 'occupation', 'zip'],
            encoding='utf-8'
        )
    except Exception as e:
        print(f"⚠️ Попередження: Не вдалося завантажити дані користувачів: {e}")
        return pd.DataFrame()


def _get_movie_avg_ratings(dataset) -> Dict[int, float]:
    """Отримує середні рейтинги всіх фільмів на основі всіх оцінок"""
    try:
        # Используем предвычисленные данные если доступны
        if precomputed and hasattr(precomputed, 'avg_ratings'):
            return precomputed.avg_ratings
        else:
            avg_ratings = dataset.ratings.groupby('movieId')['rating'].mean().to_dict()
            return avg_ratings
    except Exception as e:
        print(f"❌ Помилка отримання середніх рейтингів: {e}")
        return {}


def _get_unrated_movies_fast(user_id: int, dataset) -> np.ndarray:
    """
    ОПТИМИЗИРОВАННАЯ версия - быстро получает непросмотренные фильмы
    """
    try:
        # Используем предвычисленные данные для скорости
        if precomputed and hasattr(precomputed, 'get_unrated_movies_fast'):
            return precomputed.get_unrated_movies_fast(user_id)

        # Fallback на оригинальную логику
        all_unique_movies = np.unique(dataset.ratings['movieId'].values)
        all_unique_movies = all_unique_movies[~np.isnan(all_unique_movies)].astype(int)

        user_ratings = dataset.ratings[dataset.ratings['userId'] == user_id]
        rated_movies = set(user_ratings['movieId'].values)
        unrated_movies = np.array([m for m in all_unique_movies if m not in rated_movies])

        return unrated_movies

    except Exception as e:
        print(f"❌ Помилка в _get_unrated_movies_fast: {e}")
        return np.array([])


def get_recommendations_fast(model, user_id: int, dataset, top_k: int = 10) -> pd.DataFrame:
    """
    УСКОРЕННАЯ версия получения рекомендаций с использованием предвычислений
    """
    model.eval()

    try:
        start_time = time.time()

        # БЫСТРЫЙ поиск непросмотренных фильмов
        unrated_movies = _get_unrated_movies_fast(user_id, dataset)

        if len(unrated_movies) == 0:
            print("📝 Користувач оцінив всі доступні фільми")
            return pd.DataFrame(columns=['title', 'genres', 'real_rating', 'pred_rating'])

        print(f"🔍 Знайдено {len(unrated_movies)} непереглянутих фільмів")
        print("🔄 Генерація прогнозів...")

        # УМЕНЬШАЕМ размер батча для стабильности
        batch_size = 1000
        all_predictions = []

        total_batches = (len(unrated_movies) + batch_size - 1) // batch_size
        print(f"📦 Розбиваємо на {total_batches} батчів по {batch_size} фільмів")

        for i in range(0, len(unrated_movies), batch_size):
            batch_end = min(i + batch_size, len(unrated_movies))
            batch_movies = unrated_movies[i:batch_end]
            current_batch = (i // batch_size) + 1

            # Показываем прогресс для КАЖДОГО батча
            print(f"📦 Батч {current_batch}/{total_batches} ({len(batch_movies)} фільмів)...")

            movies_tensor = torch.tensor(batch_movies, dtype=torch.long)
            users_tensor = torch.tensor([user_id] * len(batch_movies), dtype=torch.long)

            with torch.no_grad():
                batch_predictions = model(users_tensor, movies_tensor)
                all_predictions.append(batch_predictions)

        # Объединяем все прогнозы
        if all_predictions:
            predictions = torch.cat(all_predictions)
            print(f"✅ Згенеровано {len(predictions)} прогнозів")
        else:
            predictions = torch.tensor([])
            print("⚠️ Прогнози не згенеровані")

        # Декодируем movieId обратно в оригинальные ID
        print("🔧 Декодування ідентифікаторів фільмів...")
        original_movie_ids = dataset.movie_encoder.inverse_transform(unrated_movies)

        # БЫСТРОЕ создание результатов с предвычисленными данными
        print("📊 Формування результатів...")
        results_data = []

        processed_count = 0
        total_to_process = min(len(original_movie_ids), len(predictions))

        for i, original_movie_id in enumerate(original_movie_ids):
            if i >= len(predictions):
                break

            # Показываем прогресс каждые 500 фильмов
            if i % 500 == 0:
                print(f"📊 Оброблено {i}/{total_to_process} фільмів...")

            # Используем предвычисленные данные если доступны
            if precomputed and hasattr(precomputed, 'get_movie_info_fast'):
                movie_info = precomputed.get_movie_info_fast(original_movie_id)
                title = movie_info['title']
                genres = movie_info['genres']
                real_rating = movie_info['avg_rating']
            else:
                # Fallback на оригинальную логику
                movie_row = dataset.movies[dataset.movies['movieId'] == original_movie_id]
                if not movie_row.empty:
                    title = movie_row.iloc[0]['title']
                    genres = movie_row.iloc[0]['genres']
                    real_rating = _get_movie_avg_ratings(dataset).get(original_movie_id, 0.0)
                else:
                    continue

            results_data.append({
                'title': title,
                'genres': genres,
                'real_rating': real_rating,
                'pred_rating': predictions[i].item()
            })
            processed_count += 1

        print(f"✅ Оброблено {processed_count} фільмів")

        # Создаем DataFrame и сортируем
        results_df = pd.DataFrame(results_data)

        if not results_df.empty:
            print("🎯 Сортування результатів...")
            results_df = (results_df
                          .drop_duplicates(subset=['title'])
                          .sort_values('pred_rating', ascending=False)
                          .head(top_k)
                          .reset_index(drop=True))

        total_time = time.time() - start_time
        print(f"✅ Рекомендації готові за {total_time:.2f} сек")
        print(f"🎯 Знайдено {len(results_df)} рекомендацій")

        return results_df

    except Exception as e:
        print(f"❌ Помилка в get_recommendations_fast: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['title', 'genres', 'real_rating', 'pred_rating'])


def get_recommendations_ultra_fast(model, user_id: int, dataset, top_k: int = 10) -> pd.DataFrame:
    """
    УЛЬТРА-БЫСТРАЯ версия - обрабатывает только TOP-N фильмов
    """
    model.eval()

    try:
        start_time = time.time()
        print("⚡ УЛЬТРА-ШВИДКА ОБРОБКА...")

        # Берем только 1000 самых популярных непросмотренных фильмов
        unrated_movies = _get_unrated_movies_fast(user_id, dataset)

        if len(unrated_movies) == 0:
            print("📝 Користувач оцінив всі доступні фільми")
            return pd.DataFrame(columns=['title', 'genres', 'real_rating', 'pred_rating'])

        # Ограничиваем количество обрабатываемых фильмов
        max_movies_to_process = 1000
        if len(unrated_movies) > max_movies_to_process:
            print(f"⚡ Обробляємо тільки {max_movies_to_process} найпопулярніших фільмів з {len(unrated_movies)}")
            # Берем первые max_movies_to_process фильмов (они уже отсортированы по популярности)
            unrated_movies = unrated_movies[:max_movies_to_process]
        else:
            print(f"🔍 Знайдено {len(unrated_movies)} непереглянутих фільмів")

        # Единый батч для максимальной скорости
        print("🔄 Генерація прогнозів...")
        movies_tensor = torch.tensor(unrated_movies, dtype=torch.long)
        users_tensor = torch.tensor([user_id] * len(unrated_movies), dtype=torch.long)

        with torch.no_grad():
            predictions = model(users_tensor, movies_tensor)

        print(f"✅ Згенеровано {len(predictions)} прогнозів")

        # Декодируем movieId
        original_movie_ids = dataset.movie_encoder.inverse_transform(unrated_movies)

        # Быстрое создание результатов
        results_data = []
        for i, original_movie_id in enumerate(original_movie_ids):
            if precomputed and hasattr(precomputed, 'get_movie_info_fast'):
                movie_info = precomputed.get_movie_info_fast(original_movie_id)
                results_data.append({
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'real_rating': movie_info['avg_rating'],
                    'pred_rating': predictions[i].item()
                })

        # Создаем DataFrame и сортируем
        results_df = pd.DataFrame(results_data)
        if not results_df.empty:
            results_df = (results_df
                          .drop_duplicates(subset=['title'])
                          .sort_values('pred_rating', ascending=False)
                          .head(top_k)
                          .reset_index(drop=True))

        total_time = time.time() - start_time
        print(f"⚡ Рекомендації готові за {total_time:.2f} сек")
        print(f"🎯 Знайдено {len(results_df)} рекомендацій")

        return results_df

    except Exception as e:
        print(f"❌ Помилка: {e}")
        return pd.DataFrame(columns=['title', 'genres', 'real_rating', 'pred_rating'])


def get_recommendations(model, user_id: int, dataset, top_k: int = 10) -> pd.DataFrame:
    """
    Основная функция - теперь использует быструю версию по умолчанию
    """
    return get_recommendations_fast(model, user_id, dataset, top_k)


def get_recommendations_filtered_fast(model, user_id: int, dataset, top_k: int = 10,
                                      genre: Optional[str] = None, max_age: Optional[int] = None,
                                      gender: Optional[str] = None) -> pd.DataFrame:
    """
    Ускоренная версия с фильтрами
    """
    # Сначала получаем все рекомендации быстро
    print("🔍 Отримання рекомендацій...")
    all_recommendations = get_recommendations_fast(model, user_id, dataset, top_k=1000)

    if all_recommendations.empty:
        return all_recommendations

    initial_count = len(all_recommendations)

    # Применяем фильтры
    if genre:
        mask = all_recommendations['genres'].str.contains(genre, case=False, na=False)
        all_recommendations = all_recommendations[mask]
        filtered_count = len(all_recommendations)
        print(f"🎭 Після фільтра за жанром '{genre}': {filtered_count} з {initial_count} фільмів")

    # Возрастные фильтры
    if max_age is not None:
        all_recommendations = _apply_age_based_filters(all_recommendations, max_age)

    # Фильтр по полу пользователя
    if gender is not None:
        user_passed_filter = _apply_user_filters(user_id, dataset, None, gender)
        if not user_passed_filter:
            print("❌ Користувач не пройшов фільтр за статтю")
            return pd.DataFrame(columns=['title', 'genres', 'real_rating', 'pred_rating'])

    # Возвращаем топ-K после фильтрации
    final_count = min(top_k, len(all_recommendations))
    print(f"✅ Залишилось {final_count} рекомендацій після фільтрів")
    return all_recommendations.head(top_k).reset_index(drop=True)


def get_recommendations_filtered(model, user_id: int, dataset, top_k: int = 10,
                                 genre: Optional[str] = None, max_age: Optional[int] = None,
                                 gender: Optional[str] = None) -> pd.DataFrame:
    """
    Основная функция с фильтрами - теперь использует быструю версию
    """
    return get_recommendations_filtered_fast(model, user_id, dataset, top_k, genre, max_age, gender)


def _apply_user_filters(user_id: int, dataset, max_age: Optional[int], gender: Optional[str]) -> bool:
    """Застосовує фільтри за віком та статтю користувача"""
    try:
        user_info = _get_user_info(user_id, dataset)

        if user_info.empty:
            print("⚠️ Не вдалося отримати інформацію про користувача")
            return True

        user_age = user_info['age'].values[0]
        user_gender = user_info['gender'].values[0]

        original_user_id = dataset.user_encoder.inverse_transform([user_id])[0]

        print(f"👤 Користувач {original_user_id}: вік {user_age}, стать {user_gender}")

        filter_failed = False

        if max_age is not None:
            print(f"🎯 Будуть рекомендовані фільми для віку до {max_age} років")

        if gender is not None:
            if user_gender != gender.upper():
                print(f"🚫 Стать користувача не збігається ({user_gender} != {gender.upper()})")
                filter_failed = True
            else:
                print(f"✅ Користувач відповідає фільтру статі ({user_gender} == {gender.upper()})")

        if not filter_failed:
            print("✅ Фільтри успішно пройдені")
            return True
        else:
            print("❌ Користувач не пройшов фільтри")
            return False

    except Exception as e:
        print(f"⚠️ Попередження при застосуванні фільтрів: {e}")
        return True


def _apply_age_based_filters(results_df: pd.DataFrame, max_age: Optional[int]) -> pd.DataFrame:
    """
    Застосовує фільтри на основі віку для рекомендацій
    """
    if max_age is None:
        return results_df

    print(f"🎯 Застосування вікових фільтрів для віку до {max_age} років")
    initial_count = len(results_df)

    if max_age <= 13:
        child_friendly_mask = results_df['genres'].str.contains("Children's", case=False, na=False)
        results_df = results_df[child_friendly_mask]
        print(f"👶 Для віку {max_age}: залишилось {len(results_df)} дитячих фільмів з {initial_count}")
    elif max_age <= 18:
        teen_safe_mask = ~results_df['genres'].str.contains("Horror|Thriller|Crime|Film-Noir", case=False, na=False)
        results_df = results_df[teen_safe_mask]
        print(f"🧒 Для віку {max_age}: залишилось {len(results_df)} безпечних фільмів з {initial_count}")

    return results_df


def _get_user_info(user_id: int, dataset) -> pd.DataFrame:
    """Отримує інформацію про користувача з кешуванням"""
    try:
        original_user_id = dataset.user_encoder.inverse_transform([user_id])[0]

        if hasattr(dataset, 'users') and dataset.users is not None:
            user_info = dataset.users[dataset.users['userId'] == original_user_id]
            if not user_info.empty:
                return user_info

        users_path = r"C:\Files\uni\NeutralCF\data\users_ascii.dat"
        users_df = _load_users_data(users_path)
        user_info = users_df[users_df['userId'] == original_user_id]

        return user_info

    except Exception as e:
        print(f"❌ Помилка отримання інформації про користувача {user_id}: {e}")
        return pd.DataFrame()


def get_user_stats(user_id: int, dataset) -> Dict:
    """
    Отримує статистику користувача для аналізу
    """
    try:
        original_user_id = dataset.user_encoder.inverse_transform([user_id])[0]
        user_ratings = dataset.ratings[dataset.ratings['userId'] == user_id]

        stats = {
            'user_id': original_user_id,
            'total_ratings': len(user_ratings) if len(user_ratings) > 0 else 0,
            'average_rating': round(user_ratings['rating'].mean(), 2) if len(user_ratings) > 0 else 0,
            'rated_movies_count': len(user_ratings['movieId'].unique()) if len(user_ratings) > 0 else 0,
        }

        if len(user_ratings) > 0:
            stats['rating_std'] = round(user_ratings['rating'].std(), 2)
        else:
            stats['rating_std'] = 0

        user_info = _get_user_info(user_id, dataset)
        if not user_info.empty:
            stats.update({
                'age': user_info['age'].values[0],
                'gender': user_info['gender'].values[0],
                'occupation': user_info['occupation'].values[0]
            })

        return stats

    except Exception as e:
        print(f"❌ Помилка отримання статистики користувача {user_id}: {e}")
        return {
            'user_id': user_id,
            'total_ratings': 0,
            'average_rating': 0,
            'rated_movies_count': 0,
            'rating_std': 0
        }


def display_recommendations_with_scores(recommendations: pd.DataFrame, user_id: int) -> None:
    """Покращене відображення рекомендацій з реальними рейтингами"""
    if recommendations.empty:
        print("😞 Рекомендації не знайдені")
        return

    print(f"\n🎬 ТОП-{len(recommendations)} РЕКОМЕНДАЦІЙ ДЛЯ КОРИСТУВАЧА {user_id}")
    print("=" * 80)

    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        real_score = row['real_rating']
        pred_score = row['pred_rating']

        stars = "⭐" * min(5, int(round(real_score)))
        print(f"{i:2d}. {row['title'][:50]:50} | Реальний рейтинг: {real_score:.2f} {stars}")
        print(f"    Жанр: {row['genres']}")
        print(f"    Прогноз моделі: {pred_score:.2f}")
        print()