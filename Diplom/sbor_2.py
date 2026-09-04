import pandas as pd
import numpy as np

INPUT_FILE = "/Users/ulianaalekseeva/PyCharmMiscProject/brazil_income_merged.csv"
OUTPUT_FILE = "/Users/ulianaalekseeva/PyCharmMiscProject/brazil_income_merged_filled.csv"


def fill_block_with_neighbor_mean(series: pd.Series) -> pd.Series:
    s = series.copy().astype(float)
    values = s.to_numpy()
    n = len(values)

    i = 0
    while i < n:
        if pd.isna(values[i]):
            start = i
            while i < n and pd.isna(values[i]):
                i += 1
            end = i - 1

            left = start - 1
            right = i

            if left >= 0 and right < n:
                if pd.notna(values[left]) and pd.notna(values[right]):
                    fill_value = (values[left] + values[right]) / 2.0
                    values[start:right] = fill_value
        else:
            i += 1

    return pd.Series(values, index=s.index)


# читаем файл
df = pd.read_csv(INPUT_FILE)

# приведение типов
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# сортировка
df = df.sort_values(["tax_type", "decile", "year"]).reset_index(drop=True)

# заполняем по каждому ряду отдельно
filled_parts = []

for (tax_type, decile), group in df.groupby(["tax_type", "decile"], sort=False):
    g = group.copy().sort_values("year").reset_index(drop=True)
    g["value"] = fill_block_with_neighbor_mean(g["value"])
    filled_parts.append(g)

result = pd.concat(filled_parts, ignore_index=True)
result = result.sort_values(["tax_type", "decile", "year"]).reset_index(drop=True)

# сохраняем
result.to_csv(OUTPUT_FILE, index=False)

print("Готово.")
print("Сохранено в:", OUTPUT_FILE)