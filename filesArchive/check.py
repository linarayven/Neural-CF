with open(r"/data/movies_ascii.dat", "r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f, 1):
        if "�" in line:
            print(f"Строка {i}: {line.strip()}")
