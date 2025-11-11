from pathlib import Path
from unidecode import unidecode

ratings_in = Path(r"/data/oldData/ratings.dat")
ratings_out = Path(r"/data/ratings_ascii.dat")

with ratings_in.open("r", errors="replace") as f_in, ratings_out.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        f_out.write(unidecode(line))

print(f"Готово! Файл с ASCII-символами сохранён как {ratings_out}")
