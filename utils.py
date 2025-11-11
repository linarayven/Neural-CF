import torch
import pandas as pd
import numpy as np
import os
from functools import lru_cache
from typing import Dict, List, Optional, Union


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
        avg_ratings = dataset.ratings.groupby('movieId')['rating'].mean().to_dict()
        return avg_ratings
    except Exception as e:
        print(f"❌ Помилка отримання середніх рейтингів: {e}")
        return {}


def _get_unrated_movies_batch(model, user_id: int, dataset, batch_size: int = 5000) -> tuple:
    """
    Оптимізоване отримання прогнозів для непереглянутих фільмів
    """
    try:
        # Отримуємо всі унікальні закодовані movieId
        all_unique_movies = np.unique(dataset.ratings['movieId'].values)

        # Фільтруємо NaN значення
        all_unique_movies = all_unique_movies[~np.isnan(all_unique_movies)].astype(int)

        # Отримуємо фільми, які користувач вже оцінив
        user_ratings = dataset.ratings[dataset.ratings['userId'] == user_id]
        rated_movies = set(user_ratings['movieId'].values)

        # Знаходимо фільми, які користувач ще не оцінив
        unrated_movies = np.array([m for m in all_unique_movies if m not in rated_movies])

        if len(unrated_movies) == 0:
            return torch.tensor([]), unrated_movies, np.array([])

        print(f"🔍 Обробка {len(unrated_movies)} непереглянутих фільмів...")

        # Обчислення прогнозів батчами для оптимізації пам'яті
        all_predictions = []

        for i in range(0, len(unrated_movies), batch_size):
            batch_movies = unrated_movies[i:i + batch_size]

            # Переконуємося, що всі значення цілі числа
            batch_movies = batch_movies.astype(int)

            if len(batch_movies) == 0:
                continue

            # Створюємо тензори для батча
            movies_tensor = torch.tensor(batch_movies, dtype=torch.long)
            users_tensor = torch.tensor([user_id] * len(batch_movies), dtype=torch.long)

            with torch.no_grad():
                batch_predictions = model(users_tensor, movies_tensor)
                all_predictions.append(batch_predictions)

        # Об'єднуємо всі прогнози
        predictions = torch.cat(all_predictions) if all_predictions else torch.tensor([])

        # Декодуємо movieId
        original_movie_ids = dataset.movie_encoder.inverse_transform(unrated_movies)

        return predictions, unrated_movies, original_movie_ids

    except Exception as e:
        print(f"❌ Помилка в _get_unrated_movies_batch: {e}")
        return torch.tensor([]), np.array([]), np.array([])


def get_recommendations(model, user_id: int, dataset, top_k: int = 10) -> pd.DataFrame:
    """
    Отримує топ-K рекомендацій для користувача, показуючи реальні середні рейтинги фільмів

    Args:
        model: Навчена Neural CF модель
        user_id: ID користувача (закодоване)
        dataset: Об'єкт MovieDataset
        top_k: Кількість рекомендацій для повернення

    Returns:
        DataFrame з топ-K рекомендацій
    """
    model.eval()

    try:
        predictions, unrated_movies, original_movie_ids = _get_unrated_movies_batch(model, user_id, dataset)

        if len(unrated_movies) == 0:
            print("📝 Користувач оцінив всі доступні фільми")
            return pd.DataFrame(columns=['title', 'genres', 'real_rating', 'pred_rating'])

        if len(predictions) == 0:
            print("⚠️ Не вдалося отримати прогнози")
            return pd.DataFrame(columns=['title', 'genres', 'real_rating', 'pred_rating'])

        print(f"✅ Отримано прогнози для {len(unrated_movies)} фільмів")

        # Отримуємо реальні середні рейтинги всіх фільмів
        avg_ratings = _get_movie_avg_ratings(dataset)

        # Створюємо тимчасовий DataFrame для об'єднання
        temp_df = pd.DataFrame({
            'movieId': original_movie_ids,
            'pred_rating': predictions.numpy()
        })

        # Додаємо реальні рейтинги
        temp_df['real_rating'] = temp_df['movieId'].map(avg_ratings)

        # Об'єднуємо з інформацією про фільми одним запитом
        movie_info = dataset.movies[dataset.movies['movieId'].isin(original_movie_ids)].copy()
        results_df = movie_info.merge(temp_df, on='movieId', how='inner')

        # Сортуємо за прогнозом моделі, але показуємо реальний рейтинг
        results_df = (results_df
        .drop_duplicates(subset=['title'])
        .sort_values('pred_rating', ascending=False)
        .head(top_k)
        [['title', 'genres', 'real_rating', 'pred_rating']])

        print(f"🎯 Повернено {len(results_df)} рекомендацій")
        return results_df.reset_index(drop=True)

    except Exception as e:
        print(f"❌ Помилка в get_recommendations: {e}")
        return pd.DataFrame(columns=['title', 'genres', 'real_rating', 'pred_rating'])


def _apply_user_filters(user_id: int, dataset, max_age: Optional[int], gender: Optional[str]) -> bool:
    """Застосовує фільтри за віком та статтю користувача"""
    try:
        user_info = _get_user_info(user_id, dataset)

        if user_info.empty:
            print("⚠️ Не вдалося отримати інформацію про користувача")
            return True  # Продовжуємо без фільтрів, якщо немає даних

        user_age = user_info['age'].values[0]
        user_gender = user_info['gender'].values[0]

        # Отримуємо оригінальний ID для відображення
        original_user_id = dataset.user_encoder.inverse_transform([user_id])[0]

        print(f"👤 Користувач {original_user_id}: вік {user_age}, стать {user_gender}")

        # Перевіряємо фільтри
        filter_failed = False

        if max_age is not None:
            # max_age - це вік для якого рекомендуємо, а не обмеження користувача
            print(f"🎯 Будуть рекомендовані фільми для віку до {max_age} років")

        if gender is not None:
            # Фільтр: стать користувача повинна збігатися
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
        return True  # Продовжуємо у разі помилки


def _apply_age_based_filters(results_df: pd.DataFrame, max_age: Optional[int]) -> pd.DataFrame:
    """
    Застосовує фільтри на основі віку для рекомендацій

    Args:
        results_df: DataFrame з рекомендаціями
        max_age: Максимальний вік для якого рекомендуємо

    Returns:
        Відфільтрований DataFrame
    """
    if max_age is None:
        return results_df

    print(f"🎯 Застосування вікових фільтрів для віку до {max_age} років")

    initial_count = len(results_df)

    if max_age <= 13:
        # Для дітей до 13 років - ТІЛЬКИ дитячі фільми (Children's)
        child_friendly_mask = results_df['genres'].str.contains("Children's", case=False, na=False)
        results_df = results_df[child_friendly_mask]
        print(f"👶 Для віку {max_age}: залишилось {len(results_df)} дитячих фільмів з {initial_count}")

    elif max_age <= 18:
        # Для підлітків - виключаємо екстремальний контент
        teen_safe_mask = ~results_df['genres'].str.contains("Horror|Thriller|Crime|Film-Noir", case=False, na=False)
        results_df = results_df[teen_safe_mask]
        print(f"🧒 Для віку {max_age}: залишилось {len(results_df)} безпечних фільмів з {initial_count}")

    return results_df


def get_recommendations_filtered(model, user_id: int, dataset, top_k: int = 10,
                                 genre: Optional[str] = None, max_age: Optional[int] = None,
                                 gender: Optional[str] = None) -> pd.DataFrame:
    """
    Отримує топ-K рекомендацій з фільтрацією за жанром, віком та статтю
    """
    model.eval()

    try:
        print(f"🎯 Застосування фільтрів: max_age={max_age}, gender={gender}, genre={genre}")

        # Спочатку застосовуємо фільтри за статтю
        if gender is not None:
            user_passed_filter = _apply_user_filters(user_id, dataset, None, gender)
            if not user_passed_filter:
                print("❌ Користувач не пройшов фільтр за статтю")
                return pd.DataFrame(columns=['title', 'genres', 'real_rating', 'pred_rating'])
        else:
            print("ℹ️ Фільтр за статтю не застосовується")

        # Отримуємо прогнози
        predictions, unrated_movies, original_movie_ids = _get_unrated_movies_batch(model, user_id, dataset)

        if len(unrated_movies) == 0:
            print("📝 Користувач оцінив всі доступні фільми")
            return pd.DataFrame(columns=['title', 'genres', 'real_rating', 'pred_rating'])

        # Отримуємо реальні середні рейтинги всіх фільмів
        avg_ratings = _get_movie_avg_ratings(dataset)

        # Створюємо тимчасовий DataFrame
        temp_df = pd.DataFrame({
            'movieId': original_movie_ids,
            'pred_rating': predictions.numpy()
        })

        # Додаємо реальні рейтинги
        temp_df['real_rating'] = temp_df['movieId'].map(avg_ratings)

        # Об'єднуємо з інформацію про фільми
        movie_info = dataset.movies[dataset.movies['movieId'].isin(original_movie_ids)].copy()
        results_df = movie_info.merge(temp_df, on='movieId', how='inner')

        # Застосовуємо фільтр за жанром
        if genre:
            initial_count = len(results_df)
            mask = results_df['genres'].str.contains(genre, case=False, na=False, regex=False)
            results_df = results_df[mask]
            filtered_count = len(results_df)
            print(f"🎭 Після фільтра за жанром '{genre}': {filtered_count} з {initial_count} фільмів")

            if filtered_count == 0:
                print(f"😞 Жоден фільм не відповідає жанру '{genre}'")
                return pd.DataFrame(columns=['title', 'genres', 'real_rating', 'pred_rating'])
        else:
            print("ℹ️ Фільтр за жанром не застосовується")

        # Застосовуємо вікові фільтри
        if max_age is not None:
            results_df = _apply_age_based_filters(results_df, max_age)

        # Фінальна обробка
        if len(results_df) > 0:
            results_df = (results_df
            .drop_duplicates(subset=['title'])
            .sort_values('pred_rating', ascending=False)
            .head(top_k)
            [['title', 'genres', 'real_rating', 'pred_rating']])

            print(f"🎯 Повернено {len(results_df)} рекомендацій після фільтрації")
        else:
            print("😞 Немає підходящих рекомендацій після фільтрації")

        return results_df.reset_index(drop=True)

    except Exception as e:
        print(f"❌ Помилка в get_recommendations_filtered: {e}")
        return pd.DataFrame(columns=['title', 'genres', 'real_rating', 'pred_rating'])


def _get_user_info(user_id: int, dataset) -> pd.DataFrame:
    """Отримує інформацію про користувача з кешуванням"""
    try:
        # Декодуємо ID користувача назад до оригінального
        original_user_id = dataset.user_encoder.inverse_transform([user_id])[0]

        # Спочатку перевіряємо dataset
        if hasattr(dataset, 'users') and dataset.users is not None:
            user_info = dataset.users[dataset.users['userId'] == original_user_id]
            if not user_info.empty:
                return user_info

        # Fallback: завантажуємо з файлу
        users_path = r"C:\Files\uni\NeutralCF\data\users_ascii.dat"
        users_df = _load_users_data(users_path)
        user_info = users_df[users_df['userId'] == original_user_id]

        if not user_info.empty:
            print(f"✅ Інформація про користувача {original_user_id} знайдена")
        else:
            print(f"❌ Інформація про користувача {original_user_id} не знайдена")

        return user_info

    except Exception as e:
        print(f"❌ Помилка отримання інформації про користувача {user_id}: {e}")
        return pd.DataFrame()


def get_user_stats(user_id: int, dataset) -> Dict:
    """
    Отримує статистику користувача для аналізу
    """
    try:
        # Отримуємо оригінальний ID користувача для відображення
        original_user_id = dataset.user_encoder.inverse_transform([user_id])[0]

        user_ratings = dataset.ratings[dataset.ratings['userId'] == user_id]

        stats = {
            'user_id': original_user_id,  # Показуємо оригінальний ID
            'total_ratings': len(user_ratings) if len(user_ratings) > 0 else 0,
            'average_rating': round(user_ratings['rating'].mean(), 2) if len(user_ratings) > 0 else 0,
            'rated_movies_count': len(user_ratings['movieId'].unique()) if len(user_ratings) > 0 else 0,
        }

        if len(user_ratings) > 0:
            stats['rating_std'] = round(user_ratings['rating'].std(), 2)
        else:
            stats['rating_std'] = 0

        # Додаємо інформацію про користувача
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

        # Візуальне представлення реального рейтингу
        stars = "⭐" * min(5, int(round(real_score)))
        print(f"{i:2d}. {row['title'][:50]:50} | Реальний рейтинг: {real_score:.2f} {stars}")
        print(f"    Жанр: {row['genres']}")
        print(f"    Прогноз моделі: {pred_score:.2f}")
        print()