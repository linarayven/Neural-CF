import pandas as pd


def show_all_genres():
    """Показує всі доступні жанри фільмів"""
    MOVIES_PATH = r"C:\Files\uni\NeutralCF\data\movies_ascii.dat"

    # Завантажуємо фільми
    movies_df = pd.read_csv(
        MOVIES_PATH,
        sep='::',
        engine='python',
        names=['movieId', 'title', 'genres'],
        encoding='utf-8'
    )

    # Збираємо всі унікальні жанри
    all_genres = set()
    for genres in movies_df['genres']:
        if pd.notna(genres):
            for genre in str(genres).split('|'):
                all_genres.add(genre)

    # Виводимо відсортований список
    print("🎬 ВСІ ДОСТУПНІ ЖАНРИ:")
    print("-" * 25)
    for genre in sorted(all_genres):
        print(f"  {genre}")

    print(f"\n📊 Всього жанрів: {len(all_genres)}")


if __name__ == "__main__":
    show_all_genres()