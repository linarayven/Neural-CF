# config.py
"""
Конфигурация для оптимизации системы рекомендаций
"""

# Настройки оптимизации
OPTIMIZATION_CONFIG = {
    'batch_size': 10000,           # Размер батча для предсказаний
    'cache_enabled': True,         # Включить кеширование
    'precomputed_path': 'precomputed_cache.pkl',
    'max_recommendations': 1000,   # Максимум рекомендаций для поиска
    'use_fast_functions': True,    # Использовать быстрые функции
    'enable_progress_bars': True   # Показывать прогресс-бары
}

# Пути к данным
DATA_PATHS = {
    'ratings': r"C:\Files\uni\NeutralCF\data\ratings_ascii.dat",
    'movies': r"C:\Files\uni\NeutralCF\data\movies_ascii.dat",
    'users': r"C:\Files\uni\NeutralCF\data\users_ascii.dat",
    'model': r"C:\Files\uni\NeutralCF\neural_cf_model.pth"
}

# Настройки модели
MODEL_CONFIG = {
    'embedding_size': 32,
    'hidden_sizes': [64, 32],
    'dropout_rate': 0.1
}

# Настройки интерфейса
UI_CONFIG = {
    'default_top_k': 10,
    'max_display_users': 20,
    'show_timing': True
}

def get_config():
    """Возвращает конфигурацию"""
    return {
        'optimization': OPTIMIZATION_CONFIG,
        'paths': DATA_PATHS,
        'model': MODEL_CONFIG,
        'ui': UI_CONFIG
    }