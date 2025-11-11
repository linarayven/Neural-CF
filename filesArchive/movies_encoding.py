from unidecode import unidecode
from pathlib import Path

input_file = Path(r"C:\Files\uni\NeutralCF\data\ascii.dat")
output_file = Path(r"/data/movies_ascii.dat")

with input_file.open("r", encoding="cp1252", errors="ignore") as f_in, \
     output_file.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        f_out.write(unidecode(line))

print(f"Готово! Файл с ASCII-символами сохранён как {output_file}")
