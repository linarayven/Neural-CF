# test_recommendations.py
import torch
import pandas as pd
import numpy as np
import os
import time
from src.dataset import get_dataloader
from src.model import NeuralCF
from src.utils import get_recommendations, get_recommendations_filtered, get_user_stats, \
    display_recommendations_with_scores, get_recommendations_fast, get_recommendations_ultra_fast


class InteractiveRecommender:
    def __init__(self):
        # Шляхи до даних
        self.RATINGS_PATH = r"C:\Files\uni\NeutralCF\data\ratings_ascii.dat"
        self.MOVIES_PATH = r"C:\Files\uni\NeutralCF\data\movies_ascii.dat"
        self.USERS_PATH = r"C:\Files\uni\NeutralCF\data\users_ascii.dat"
        self.MODEL_PATH = r"C:\Files\uni\NeutralCF\neural_cf_model.pth"

        # Ініціалізація змінних
        self.model = None
        self.dataloader = None
        self.dataset = None
        self.user_encoder = None
        self.movie_encoder = None
        self.n_users = 0
        self.n_movies = 0

        # Завантаження даних та моделі
        self.load_data_and_model_optimized()

    def load_data_and_model_optimized(self):
        """Оптимизированная загрузка с предвычислениями"""
        print("=== Завантаження даних та моделі ===")

        try:
            # Проверка файлов
            for path in [self.RATINGS_PATH, self.MOVIES_PATH, self.USERS_PATH, self.MODEL_PATH]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"❌ Файл не знайдено: {path}")

            # Загрузка данных
            self.dataloader, self.n_users, self.n_movies, self.user_encoder, self.movie_encoder, self.dataset = get_dataloader(
                self.RATINGS_PATH, self.MOVIES_PATH, self.USERS_PATH, batch_size=64
            )

            # ✅ Инициализация предвычислений
            from src.precomputed import init_precomputed
            init_precomputed(self.dataset)

            # Загрузка модели
            self.model = NeuralCF(self.n_users, self.n_movies)
            checkpoint = torch.load(self.MODEL_PATH, map_location=torch.device('cpu'))

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.eval()

            print(f"✅ Модель та попередні обчислення завантажено")
            print(f"📊 Дані: {self.n_users} користувачів, {self.n_movies} фільмів")

        except Exception as e:
            print(f"❌ Помилка завантаження: {e}")
            raise

    def test_specific_user_fast(self, user_id):
        """Быстрая версия тестирования"""
        try:
            if user_id not in self.user_encoder.classes_:
                print(f"❌ Користувача {user_id} не знайдено!")
                return

            encoded_user_id = self.user_encoder.transform([user_id])[0]

            print(f"\n=== ШВИДКЕ ТЕСТУВАННЯ ДЛЯ КОРИСТУВАЧА {user_id} ===")

            # Статистика пользователя
            stats = get_user_stats(encoded_user_id, self.dataset)
            print("\n📊 Статистика користувача:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

            # Использовать быструю версию
            start_time = time.time()
            result = get_recommendations_fast(
                self.model, encoded_user_id, self.dataset, top_k=10
            )
            prediction_time = time.time() - start_time

            print(f"⏱️  Час генерації рекомендацій: {prediction_time:.2f} сек")

            display_recommendations_with_scores(result, user_id)

            # Опция сохранения
            self._save_recommendations_option(result, user_id)

        except Exception as e:
            print(f"❌ Помилка швидкого тестування: {e}")
            import traceback
            traceback.print_exc()

    def test_specific_user_ultra_fast(self, user_id):
        """УЛЬТРА-быстрая версия тестирования"""
        try:
            if user_id not in self.user_encoder.classes_:
                print(f"❌ Користувача {user_id} не знайдено!")
                return

            encoded_user_id = self.user_encoder.transform([user_id])[0]

            print(f"\n=== УЛЬТРА-ШВИДКЕ ТЕСТУВАННЯ ДЛЯ КОРИСТУВАЧА {user_id} ===")

            # Статистика пользователя
            stats = get_user_stats(encoded_user_id, self.dataset)
            print("\n📊 Статистика користувача:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

            # Использовать УЛЬТРА-быструю версию
            start_time = time.time()
            result = get_recommendations_ultra_fast(
                self.model, encoded_user_id, self.dataset, top_k=10
            )
            prediction_time = time.time() - start_time

            print(f"⏱️  Час генерації рекомендацій: {prediction_time:.2f} сек")

            display_recommendations_with_scores(result, user_id)

            # Опция сохранения
            self._save_recommendations_option(result, user_id)

        except Exception as e:
            print(f"❌ Помилка ультра-швидкого тестування: {e}")

    def clear_cache_command(self):
        """Очистка кеша (если данные изменились)"""
        from src.precomputed import clear_cache
        clear_cache()
        print("🗑️  Кеш очищено. При наступному запуску дані будуть переобчислені.")

    # ОРИГИНАЛЬНЫЕ МЕТОДЫ (остаются без изменений)
    def _load_users_backup(self):
        """Резервне завантаження даних користувачів"""
        try:
            print("🔧 Завантаження даних користувачів з файлу...")
            users_df = pd.read_csv(
                self.USERS_PATH,
                sep='::',
                engine='python',
                names=['userId', 'gender', 'age', 'occupation', 'zip'],
                encoding='utf-8'
            )
            self.dataset.users = users_df
            print(f"✅ Резервне завантаження успішне: {len(users_df)} користувачів")
        except Exception as e:
            print(f"❌ Помилка резервного завантаження: {e}")
            self.dataset.users = None

    def find_users_by_criteria(self, age=None, gender=None, occupation=None):
        """Знаходить користувачів за критеріями"""
        try:
            if not hasattr(self.dataset, 'users') or self.dataset.users is not None:
                print("⚠️ Інформація про користувачів недоступна")
                return []

            users_df = self.dataset.users
            mask = pd.Series([True] * len(users_df))

            if age is not None:
                mask &= (users_df['age'] == age)
            if gender is not None:
                mask &= (users_df['gender'] == gender.upper())
            if occupation is not None:
                mask &= (users_df['occupation'] == occupation)

            matching_users = users_df[mask]
            print(f"✅ Знайдено {len(matching_users)} користувачів")
            return matching_users['userId'].tolist()

        except Exception as e:
            print(f"❌ Помилка пошуку користувачів: {e}")
            return []

    def display_user_options(self, user_list):
        """Відображає список користувачів для вибору"""
        if not user_list:
            print("😞 Користувачів не знайдено!")
            return None

        print(f"\n🔍 Знайдено {len(user_list)} користувачів:")
        users_info = []
        display_count = min(20, len(user_list))

        for user_id in user_list[:display_count]:
            try:
                user_info = self.dataset.users[self.dataset.users['userId'] == user_id]
                if not user_info.empty:
                    encoded_id = self.user_encoder.transform([user_id])[0]
                    stats = get_user_stats(encoded_id, self.dataset)

                    users_info.append({
                        'ID': user_id,
                        'Вік': user_info['age'].values[0],
                        'Стать': user_info['gender'].values[0],
                        'Професія': user_info['occupation'].values[0],
                        'Оцінки': stats.get('total_ratings', 'N/A'),
                    })
            except Exception:
                continue

        if users_info:
            df = pd.DataFrame(users_info)
            print(df.to_string(index=False, max_colwidth=12))
            return users_info
        return None

    def get_user_input(self):
        """Отримує критерії пошуку від користувача"""
        print("\n=== ПОШУК КОРИСТУВАЧІВ ЗА КРИТЕРІЯМИ ===")
        print("💡 Залишіть поле порожнім, щоб пропустити критерій")

        age = input("\n🔢 Вік: ").strip()
        gender = input("👥 Стать (M/F): ").strip()
        occupation = input("💼 Професія (0-20): ").strip()

        criteria = {}
        if age:
            try:
                criteria['age'] = int(age)
            except ValueError:
                print("❌ Невірний формат віку!")
                return self.get_user_input()
        if gender:
            if gender.upper() in ['M', 'F']:
                criteria['gender'] = gender.upper()
            else:
                print("❌ Стать має бути M або F!")
                return self.get_user_input()
        if occupation:
            try:
                criteria['occupation'] = int(occupation)
            except ValueError:
                print("❌ Невірний формат професії!")
                return self.get_user_input()

        if not criteria:
            print("❌ Будь ласка, введіть хоча б один критерій!")
            return self.get_user_input()

        return criteria

    def test_specific_user(self, user_id):
        """Тестує рекомендації для конкретного користувача"""
        try:
            if user_id not in self.user_encoder.classes_:
                print(f"❌ Користувача {user_id} не знайдено!")
                return

            encoded_user_id = self.user_encoder.transform([user_id])[0]
            print(f"\n=== ТЕСТУВАННЯ ДЛЯ КОРИСТУВАЧА {user_id} ===")

            # Статистика користувача
            stats = get_user_stats(encoded_user_id, self.dataset)
            print("\n📊 Статистика користувача:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

            # Получаем рекомендации БЫСТРО
            start_time = time.time()
            result = get_recommendations_fast(
                self.model, encoded_user_id, self.dataset, top_k=10
            )
            prediction_time = time.time() - start_time

            print(f"⏱️  Час генерації рекомендацій: {prediction_time:.2f} сек")

            display_recommendations_with_scores(result, user_id)
            self._save_recommendations_option(result, user_id)

        except Exception as e:
            print(f"❌ Помилка тестування користувача: {e}")

    def _save_recommendations_option(self, recommendations, user_id):
        """Запропонувати зберегти рекомендації у файл"""
        if recommendations.empty:
            return

        save_choice = input("\n💾 Зберегти рекомендації у файл? (y/N): ").strip().lower()
        if save_choice in ['y', 'yes']:
            try:
                filename = f"recommendations_user_{user_id}.csv"
                recommendations.to_csv(filename, index=False, encoding='utf-8')
                print(f"✅ Рекомендації збережено у файл: {filename}")
            except Exception as e:
                print(f"❌ Помилка збереження: {e}")

    def search_and_test_users(self):
        """Пошук користувачів за критеріями та тестування"""
        criteria = self.get_user_input()
        if not criteria:
            return

        user_list = self.find_users_by_criteria(**criteria)
        users_info = self.display_user_options(user_list)

        if users_info:
            try:
                user_choice = input("\n🎯 Оберіть ID користувача для тестування: ").strip()
                if user_choice:
                    user_id = int(user_choice)
                    found_ids = [u['ID'] for u in users_info]
                    if user_id in found_ids:
                        self.test_specific_user(user_id)
                    else:
                        print(f"❌ Користувача з ID {user_id} не знайдено!")
                else:
                    print("ℹ️ Тестування скасовано")
            except ValueError:
                print("❌ Невірний формат ID!")

    def test_by_user_id(self):
        """Тестування за конкретным ID користувача"""
        try:
            user_id = input("🔢 Введіть ID користувача: ").strip()
            if user_id:
                self.test_specific_user(int(user_id))
            else:
                print("ℹ️ Тестування скасовано")
        except ValueError:
            print("❌ Невірний формат ID!")

    def quick_test(self):
        """Швидкий тест для користувача 5"""
        try:
            if 5 in self.user_encoder.classes_:
                print("🚀 Запуск швидкого тесту для користувача 5...")
                self.test_specific_user(5)
            else:
                print("❌ Користувача 5 не знайдено!")
        except Exception as e:
            print(f"❌ Помилка швидкого тесту: {e}")

    def show_statistics(self):
        """Показує статистику даних"""
        print("\n=== СТАТИСТИКА ДАНИХ ===")
        print(f"👥 Користувачів: {self.n_users}")
        print(f"🎬 Фільмів: {self.n_movies}")
        print(f"⭐ Оцінок: {len(self.dataset.ratings):,}")

    def run(self):
        """Головний цикл програми"""
        print("🎬 ІНТЕРАКТИВНА СИСТЕМА РЕКОМЕНДАЦІЙ ФІЛЬМІВ (ОПТИМІЗОВАНА)")
        print("=" * 50)

        while True:
            print("\n📋 Головне меню:")
            print("1. 🔍 Знайти користувачів за критеріями")
            print("2. 🎯 Тестувати конкретного користувача (за ID)")
            print("3. 🚀 ШВИДКИЙ тест (користувач 5)")
            print("4. 📊 Статистика даних")
            print("5. ⚡ Надзвичайно швидкий тест")
            print("6. 🚀 УЛЬТРА-швидкий тест")
            print("7. 🗑️  Очистити кеш")
            print("8. ❌ Вийти")

            choice = input("\n🎯 Оберіть опцію (1-8): ").strip()

            if choice == '1':
                self.search_and_test_users()
            elif choice == '2':
                self.test_by_user_id()
            elif choice == '3':
                self.quick_test()
            elif choice == '4':
                self.show_statistics()
            elif choice == '5':
                user_id = input("🔢 Введіть ID для швидкого тесту: ").strip()
                if user_id:
                    self.test_specific_user_fast(int(user_id))
            elif choice == '6':
                user_id = input("🔢 Введіть ID для ультра-швидкого тесту: ").strip()
                if user_id:
                    self.test_specific_user_ultra_fast(int(user_id))
            elif choice == '7':
                self.clear_cache_command()
            elif choice == '8':
                print("👋 Дякую за використання!")
                break
            else:
                print("❌ Невірний вибір!")


def main():
    """Головна функція"""
    try:
        print("🚀 Запуск системи рекомендацій...")
        recommender = InteractiveRecommender()
        recommender.run()
    except Exception as e:
        print(f"💥 Критична помилка: {e}")
        input("Натисніть Enter для виходу...")


if __name__ == "__main__":
    main()