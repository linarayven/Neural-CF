from pathlib import Path
from unidecode import unidecode

# Пути
users_in = Path(r"/data/oldData/users")
users_out = Path(r"/data/users_ascii.dat")

# Обработка
with users_in.open("r", errors="replace") as f_in, users_out.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        f_out.write(unidecode(line))

print(f"Готово! Файл с ASCII-символами сохранён как {users_out}")
