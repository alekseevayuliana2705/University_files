import os
import pandas as pd
import numpy as np

# ==========================================================
# НАСТРОЙКИ
# ==========================================================

DATA_DIR = "/Users/ulianaalekseeva/Downloads/Brazil"
OUTPUT_DIR = "/Users/ulianaalekseeva/PyCharmMiscProject"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "brazil_income_merged.csv")

FILES = [
    ("pre", "Pre_0_50.csv"),
    ("pre", "Pre_50_100.csv"),
    ("post", "Post_0_50.csv"),
    ("post", "Post_50_100.csv"),
]

# =========================
# ЧТЕНИЕ ОДНОГО ФАЙЛА
# =========================

def load_file(path, tax_type):
    df = pd.read_csv(
        path,
        sep=";",
        skiprows=5,
        header=None,
        names=["decile", "year", "value"]
    )

    df["decile"] = df["decile"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(
        df["value"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
    df["tax_type"] = tax_type

    return df[["tax_type", "decile", "year", "value"]]

# =========================
# ОБЪЕДИНЕНИЕ
# =========================

all_frames = []

for tax_type, filename in FILES:
    full_path = os.path.join(DATA_DIR, filename)
    df = load_file(full_path, tax_type)
    all_frames.append(df)

merged = pd.concat(all_frames, ignore_index=True)

# сортировка для удобства
merged = merged.sort_values(["tax_type", "decile", "year"]).reset_index(drop=True)

# сохранение
merged.to_csv(OUTPUT_FILE, index=False)

print("Готово.")
print("Файл сохранён:", OUTPUT_FILE)