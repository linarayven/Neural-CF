import torch
import pandas as pd
import numpy as np
import os
from src.dataset import get_dataloader
from src.model import NeuralCF
from src.utils import get_recommendations, get_recommendations_filtered, get_user_stats, \
    display_recommendations_with_scores


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
        self.load_data_and_model()

    def load_data_and_model(self):
        """Завантажує дані та модель з покращеною обробкою помилок"""
        print("=== Завантаження даних та моделі ===")

        try:
            # Перевірка існування файлів
            for path in [self.RATINGS_PATH, self.MOVIES_PATH, self.USERS_PATH, self.MODEL_PATH]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"❌ Файл не знайдено: {path}")

            # Завантаження даних
            self.dataloader, self.n_users, self.n_movies, self.user_encoder, self.movie_encoder, self.dataset = get_dataloader(
                self.RATINGS_PATH, self.MOVIES_PATH, self.USERS_PATH, batch_size=64
            )

            # Діагностика: перевіряємо дані користувачів
            if hasattr(self.dataset, 'users') and self.dataset.users is not None:
                print(f"✅ Дані користувачів завантажено: {len(self.dataset.users)} записів")
                print(f"   Унікальні статі: {self.dataset.users['gender'].unique()}")
                print(f"   Розподіл за статтю:")
                gender_counts = self.dataset.users['gender'].value_counts()
                for gender, count in gender_counts.items():
                    print(f"     {gender}: {count} користувачів")
            else:
                print("⚠️ Дані користувачів не завантажено, використовуємо резервне завантаження...")
                self._load_users_backup()

            # Ініціалізація та завантаження моделі
            self.model = NeuralCF(self.n_users, self.n_movies)

            # Безпечне завантаження моделі з обробкою checkpoint
            checkpoint = torch.load(self.MODEL_PATH, map_location=torch.device('cpu'))

            # Перевіряємо тип завантажених даних
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Це checkpoint з додатковими метаданими
                    print("📦 Завантаження моделі з checkpoint...")
                    self.model.load_state_dict(checkpoint['model_state_dict'])

                    # Перевіряємо сумісність розмірів
                    if 'num_users' in checkpoint and checkpoint['num_users'] != self.n_users:
                        print(
                            f"⚠️ Попередження: Кількість користувачів у моделі ({checkpoint['num_users']}) не збігається з даними ({self.n_users})")
                    if 'num_movies' in checkpoint and checkpoint['num_movies'] != self.n_movies:
                        print(
                            f"⚠️ Попередження: Кількість фільмів у моделі ({checkpoint['num_movies']}) не збігається з даними ({self.n_movies})")

                elif 'state_dict' in checkpoint:
                    # Альтернативний формат checkpoint
                    print("📦 Завантаження моделі з state_dict...")
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Спроба завантажити як звичайний state_dict
                    print("📦 Завантаження моделі як state_dict...")
                    self.model.load_state_dict(checkpoint)
            else:
                # Неочікуваний формат
                raise ValueError("Невідомий формат файлу моделі")

            self.model.eval()

            print(f"✅ Модель завантажена: {self.n_users} користувачів, {self.n_movies} фільмів")
            print(f"✅ Дані завантажені: {len(self.dataset.ratings)} оцінок")

        except FileNotFoundError as e:
            print(f"❌ Помилка: {e}")
            raise
        except Exception as e:
            print(f"❌ Помилка завантаження моделі: {e}")
            print("💡 Спробуйте перевчити модель або перевірте формат файлу")
            raise

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
            print(f"   Унікальні статі: {users_df['gender'].unique()}")
        except Exception as e:
            print(f"❌ Помилка резервного завантаження: {e}")
            self.dataset.users = None

    def find_users_by_criteria(self, age=None, gender=None, occupation=None):
        """
        Знаходить користувачів за критеріями

        Returns:
            Список ID користувачів, що відповідають критеріям
        """
        try:
            if not hasattr(self.dataset, 'users') or self.dataset.users is None:
                print("⚠️ Інформація про користувачів недоступна")
                return []

            users_df = self.dataset.users

            # Діагностика: перевіряємо дані
            print(f"🔍 Загальна кількість користувачів у даних: {len(users_df)}")

            # Застосовуємо фільтри ТІЛЬКИ якщо значення вказано
            mask = pd.Series([True] * len(users_df))

            if age is not None:
                print(f"🔍 Фільтр за віком: {age}")
                mask &= (users_df['age'] == age)
                print(f"   Кількість після фільтра віку: {mask.sum()}")
            else:
                print("ℹ️ Фільтр за віком не застосовується")

            if gender is not None:
                print(f"🔍 Фільтр за статтю: {gender.upper()}")
                # Діагностика: перевіряємо значення статі в даних
                unique_genders = users_df['gender'].unique()
                print(f"   Унікальні значення статі в даних: {unique_genders}")

                # Застосовуємо фільтр
                gender_mask = (users_df['gender'] == gender.upper())
                print(f"   Кількість користувачів зі статтю {gender.upper()}: {gender_mask.sum()}")

                mask &= gender_mask
                print(f"   Кількість після фільтра статі: {mask.sum()}")
            else:
                print("ℹ️ Фільтр за статтю не застосовується")

            if occupation is not None:
                print(f"🔍 Фільтр за професією: {occupation}")
                mask &= (users_df['occupation'] == occupation)
                print(f"   Кількість після фільтра професії: {mask.sum()}")
            else:
                print("ℹ️ Фільтр за професією не застосовується")

            matching_users = users_df[mask]
            print(f"✅ Знайдено {len(matching_users)} користувачів за вказаними критеріями")

            # Додаткова інформація про критерії пошуку
            criteria_info = []
            if age is not None:
                criteria_info.append(f"вік={age}")
            if gender is not None:
                criteria_info.append(f"стать={gender.upper()}")
            if occupation is not None:
                criteria_info.append(f"професія={occupation}")

            if criteria_info:
                print(f"🎯 Критерії пошуку: {', '.join(criteria_info)}")

            return matching_users['userId'].tolist()

        except Exception as e:
            print(f"❌ Помилка пошуку користувачів: {e}")
            return []

    def display_user_options(self, user_list):
        """Відображає список користувачів для вибору"""
        if not user_list:
            print("😞 Користувачів не знайдено за вказаними критеріями!")
            print("💡 Спробуйте змінити критерії пошуку")
            return None

        print(f"\n🔍 Знайдено {len(user_list)} користувачів:")
        print("=" * 80)

        users_info = []
        display_count = min(20, len(user_list))

        for user_id in user_list[:display_count]:
            try:
                # Отримуємо інформацію про користувача напряму з даних
                user_info = self.dataset.users[self.dataset.users['userId'] == user_id]
                if not user_info.empty:
                    # Отримуємо статистику користувача
                    encoded_id = self.user_encoder.transform([user_id])[0]
                    stats = get_user_stats(encoded_id, self.dataset)

                    users_info.append({
                        'ID': user_id,
                        'Вік': user_info['age'].values[0],
                        'Стать': user_info['gender'].values[0],
                        'Професія': user_info['occupation'].values[0],
                        'Оцінки': stats.get('total_ratings', 'N/A'),
                        'Середній_рейтинг': stats.get('average_rating', 'N/A')
                    })
            except Exception as e:
                print(f"⚠️ Помилка обробки користувача {user_id}: {e}")
                continue

        # Виводимо таблицю
        if users_info:
            df = pd.DataFrame(users_info)
            print(df.to_string(index=False, max_colwidth=12))

            if len(user_list) > display_count:
                print(f"\n... і ще {len(user_list) - display_count} користувачів")

            return users_info
        return None

    def get_user_input(self):
        """Отримує критерії пошуку від користувача"""
        print("\n=== ПОШУК КОРИСТУВАЧІВ ЗА КРИТЕРІЯМИ ===")
        print("💡 Залишіть поле порожнім, щоб пропустити критерій")
        print("💡 Доступні значення:")
        print("   - Вік: 1, 18, 25, 35, 45, 50, 56")
        print("   - Стать: M, F")
        print("   - Професія: 0-20")

        age = input("\n🔢 Вік: ").strip()
        gender = input("👥 Стать (M/F): ").strip()
        occupation = input("💼 Професія (0-20): ").strip()

        # Конвертуємо введення
        criteria = {}

        if age:
            try:
                age_val = int(age)
                if age_val in [1, 18, 25, 35, 45, 50, 56]:
                    criteria['age'] = age_val
                else:
                    print(f"⚠️ Вік {age_val} не є стандартним значенням, але буде використано")
                    criteria['age'] = age_val
            except ValueError:
                print("❌ Невірний формат віку! Використовуйте числа.")
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
                print("❌ Невірний формат професії! Використовуйте числа 0-20.")
                return self.get_user_input()

        # Перевіряємо, чи введено хоча б один критерій
        if not criteria:
            print("❌ Будь ласка, введіть хоча б один критерій пошуку!")
            return self.get_user_input()

        return criteria

    def show_available_values(self):
        """Показує доступні значення для критеріїв пошуку"""
        if not hasattr(self.dataset, 'users') or self.dataset.users is None:
            print("❌ Інформація про користувачів недоступна")
            return

        users_df = self.dataset.users

        print("\n📊 ДОСТУПНІ ЗНАЧЕННЯ:")
        print("=" * 40)

        # Вік
        print("\n🔢 ВІК:")
        age_counts = users_df['age'].value_counts().sort_index()
        for age, count in age_counts.items():
            print(f"   {age}: {count} користувачів")

        # Стать
        print("\n👥 СТАТЬ:")
        gender_counts = users_df['gender'].value_counts()
        for gender, count in gender_counts.items():
            print(f"   {gender}: {count} користувачів")

        # Професія
        print("\n💼 ПРОФЕСІЇ:")
        occupation_counts = users_df['occupation'].value_counts().sort_index()
        for occupation, count in occupation_counts.items():
            print(f"   {occupation}: {count} користувачів")

    def get_recommendation_parameters(self):
        """Отримує параметри рекомендацій"""
        print("\n=== ПАРАМЕТРИ РЕКОМЕНДАЦІЙ ===")
        print("💡 Фільтри:")
        print("   - Жанр: фільтрує фільми за жанром")
        print("   - Вік: вказує для якого віку рекомендуємо фільми")
        print("   - Стать: фільтрує користувачів за статтю")

        try:
            top_k = input("🎯 Кількість рекомендацій (за замовчуванням 10): ").strip()
            top_k = int(top_k) if top_k else 10

            genre = input("🎭 Жанр (наприклад, Comedy, Action, Drama): ").strip()
            genre = genre if genre else None

            max_age_input = input("📅 Для якого віку рекомендуємо (наприклад, 18=для підлітків): ").strip()
            max_age = int(max_age_input) if max_age_input else None

            gender_filter = input("👥 Стать користувача для фільтра (M/F): ").strip()
            gender_filter = gender_filter.upper() if gender_filter else None

            # Додаткова інформація про фільтри
            if genre:
                print(f"ℹ️ Буде застосовано фільтр за жанром: {genre}")
            if max_age is not None:
                print(f"ℹ️ Будуть рекомендовані фільми для віку до {max_age} років")
            if gender_filter is not None:
                print(f"ℹ️ Буде застосовано фільтр: стать користувача = {gender_filter}")

            return {
                'top_k': top_k,
                'genre': genre,
                'max_age': max_age,
                'gender': gender_filter
            }
        except ValueError:
            print("❌ Невірний формат числа!")
            return self.get_recommendation_parameters()

    def test_specific_user(self, user_id):
        """Тестує рекомендації для конкретного користувача"""
        try:
            # Перевірка існування користувача
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

            # Параметри рекомендацій
            params = self.get_recommendation_parameters()

            # Отримуємо рекомендації
            if params['genre'] or params['max_age'] or params['gender']:
                print(f"\n🔍 Пошук рекомендацій з фільтрами...")
                result = get_recommendations_filtered(
                    self.model, encoded_user_id, self.dataset,
                    top_k=params['top_k'],
                    genre=params['genre'],
                    max_age=params['max_age'],
                    gender=params['gender']
                )
            else:
                print(f"\n🔍 Пошук рекомендацій...")
                result = get_recommendations(
                    self.model, encoded_user_id, self.dataset,
                    top_k=params['top_k']
                )

            # Виводимо результати
            display_recommendations_with_scores(result, user_id)

            # Запропонувати зберегти результати
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

    def run(self):
        """Головний цикл програми"""
        print("🎬 ІНТЕРАКТИВНА СИСТЕМА РЕКОМЕНДАЦІЙ ФІЛЬМІВ")
        print("=" * 50)

        while True:
            print("\n📋 Головне меню:")
            print("1. 🔍 Знайти користувачів за критеріями")
            print("2. 🎯 Тестувати конкретного користувача (за ID)")
            print("3. 🚀 Швидкий тест (користувач 5)")
            print("4. 📊 Статистика даних")
            print("5. 🔄 Безперервний режим")
            print("6. 📋 Показати доступні значення")
            print("7. ❌ Вийти")

            choice = input("\n🎯 Оберіть опцію (1-7): ").strip()

            if choice == '1':
                self.search_and_test_users()
            elif choice == '2':
                self.test_by_user_id()
            elif choice == '3':
                self.quick_test()
            elif choice == '4':
                self.show_statistics()
            elif choice == '5':
                self.continuous_mode()
            elif choice == '6':
                self.show_available_values()
            elif choice == '7':
                print("👋 Дякую за використання!")
                break
            else:
                print("❌ Невірний вибір! Спробуйте ще раз.")

    def search_and_test_users(self):
        """Пошук користувачів за критеріями та тестування"""
        criteria = self.get_user_input()

        if not criteria:
            print("❌ Будь ласка, введіть хоча б один критерій!")
            return

        # Знаходимо користувачів
        user_list = self.find_users_by_criteria(**criteria)
        users_info = self.display_user_options(user_list)

        if users_info:
            try:
                user_choice = input("\n🎯 Оберіть ID користувача для тестування (або Enter для скасування): ").strip()
                if user_choice:
                    user_id = int(user_choice)
                    # Проверяем, что выбранный ID есть в списке найденных пользователей
                    found_ids = [u['ID'] for u in users_info]
                    if user_id in found_ids:
                        self.test_specific_user(user_id)
                    else:
                        print(f"❌ Користувача з ID {user_id} не знайдено у списку!")
                        print(f"💡 Доступні ID: {found_ids[:10]}{'...' if len(found_ids) > 10 else ''}")
                else:
                    print("ℹ️ Тестування скасовано")
            except ValueError:
                print("❌ Невірний формат ID! Введіть число.")

    def test_by_user_id(self):
        """Тестування за конкретним ID користувача"""
        try:
            user_id = input("🔢 Введіть ID користувача: ").strip()
            if user_id:
                user_id = int(user_id)
                self.test_specific_user(user_id)
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

    def continuous_mode(self):
        """Режим безперервного тестування"""
        print("\n🔄 РЕЖИМ БЕЗПЕРЕРВНОГО ТЕСТУВАННЯ")
        print("💡 Введіть 'stop' для виходу")

        while True:
            user_input = input("\n🔢 Введіть ID користувача: ").strip()

            if user_input.lower() == 'stop':
                break

            try:
                user_id = int(user_input)
                self.test_specific_user(user_id)
            except ValueError:
                print("❌ Неправильний формат ID!")

    def show_statistics(self):
        """Показує статистику даних"""
        print("\n=== СТАТИСТИКА ДАНИХ ===")
        print(f"👥 Загальна кількість користувачів: {self.n_users}")
        print(f"🎬 Загальна кількість фільмів: {self.n_movies}")
        print(f"⭐ Загальна кількість оцінок: {len(self.dataset.ratings):,}")

        # Розрахунок розрідженості
        sparsity = 1 - (len(self.dataset.ratings) / (self.n_users * self.n_movies))
        print(f"📊 Розрідженість даних: {sparsity:.2%}")

        # Статистика за віком
        if hasattr(self.dataset, 'users') and self.dataset.users is not None:
            age_stats = self.dataset.users['age'].value_counts().sort_index()
            print("\n📅 Розподіл за віком:")
            for age, count in age_stats.items():
                print(f"  Вік {age}: {count} користувачів")

        # Статистика за статтю
        if hasattr(self.dataset, 'users') and self.dataset.users is not None:
            gender_stats = self.dataset.users['gender'].value_counts()
            print("\n👥 Розподіл за статтю:")
            for gender, count in gender_stats.items():
                print(f"  {gender}: {count} користувачів")

        # Статистика оцінок
        if hasattr(self.dataset, 'ratings'):
            rating_stats = self.dataset.ratings['rating'].describe()
            print(f"\n⭐ Статистика оцінок:")
            print(f"  Середнє: {rating_stats['mean']:.2f}")
            print(f"  Медіана: {rating_stats['50%']:.2f}")
            print(f"  Стандартне відхилення: {rating_stats['std']:.2f}")


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