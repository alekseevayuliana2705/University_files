import pandas as pd

FILE_PATH = "/Users/ulianaalekseeva/PyCharmMiscProject/API_FR.INR.RINR_DS2_en_csv_v2_289.csv"

# соответствие названий
COUNTRY_MAP = {
    "Russia": "Russian Federation",
    "Germany": "Germany",
    "USA": "United States",
    "UK": "United Kingdom",
    "Sweden": "Sweden",
    "Spain": "Spain",
    "Poland": "Poland",
    "Netherlands": "Netherlands",
    "France": "France",
    "China": "China",
    "Japan": "Japan",
    "Italy": "Italy",
    "Australia": "Australia",
    "Kazahstan": "Kazakhstan"
}

START_YEAR = 1980
END_YEAR = 2024

# загрузка
df = pd.read_csv(FILE_PATH, skiprows=4)

year_cols = [str(y) for y in range(START_YEAR, END_YEAR + 1)]

results = []

for my_name, wb_name in COUNTRY_MAP.items():

    row = df[df["Country Name"] == wb_name]

    if row.empty:
        print(f"⚠️ Нет данных для {my_name}")
        continue

    row = row.iloc[0]

    values = pd.to_numeric(row[year_cols], errors="coerce")

    avg_rate = values.mean(skipna=True)

    results.append({
        "country": my_name,
        "average_real_rate_percent": avg_rate,
        "average_real_rate_decimal": avg_rate / 100,
        "observations": values.notna().sum()
    })

result_df = pd.DataFrame(results)

print("\n==============================")
print("AVERAGE REAL INTEREST RATES")
print("==============================")
print(result_df.sort_values("average_real_rate_percent"))

# сохранить
result_df.to_csv("average_rates_1980_2024.csv", index=False)