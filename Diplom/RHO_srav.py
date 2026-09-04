import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================================
# SETTINGS
# ==========================================================

DATA_DIR = Path("/Users/ulianaalekseeva/PyCharmMiscProject")

COUNTRY_FILES = {
    "Russia": "phi_periods_russia.csv",
    "Germany": "phi_periods_germany.csv",
    "USA": "phi_periods_usa.csv",
    "UK": "phi_periods_uk.csv",
    "Sweden": "phi_periods_sweden.csv",
    "Spain": "phi_periods_spain.csv",
    "Poland": "phi_periods_poland.csv",
    "Netherlands": "phi_periods_netherlands.csv",
    "France": "phi_periods_france.csv",
    "China": "phi_periods_china.csv",
    "Japan": "phi_periods_japan.csv",
    "Italy": "phi_periods_italy.csv",
    "Australia": "phi_periods_australia.csv",
    "Kazahstan": "phi_periods_kazahstan.csv",
}

# Ставки r в десятичном виде
RATES = {
    "Russia": 0.145,
    "Germany": 0.0215,
    "USA": 0.0375,
    "UK": 0.0375,
    "Sweden": 0.0175,
    "Spain": 0.0215,
    "Poland": 0.0375,
    "Netherlands": 0.0215,
    "France": 0.0215,
    "China": 0.03,
    "Japan": 0.0075,
    "Italy": 0.0215,
    "Australia": 0.041,
    "Kazahstan": 0.18,
}

PERIODS = [
    "1980-1990",
    "1990-2000",
    "2000-2010",
    "2010-2020",
    "2020-2024"
]

COUNTRY_COLORS = {
    "Russia": "#d62728",
    "Germany": "#1f77b4",
    "USA": "#2ca02c",
    "UK": "#9467bd",
    "Sweden": "#17becf",
    "Spain": "#ff7f0e",
    "Poland": "#8c564b",
    "Netherlands": "#e377c2",
    "France": "#bcbd22",
    "China": "#7f7f7f",
    "Japan": "#000000",
    "Italy": "#aec7e8",
    "Australia": "#98df8a",
    "Kazahstan": "#ff9896",
}

PERIOD_STYLES = {
    "1980-1990": "-",
    "1990-2000": "--",
    "2000-2010": "-.",
    "2010-2020": ":",
    "2020-2024": (0, (5, 1)),
}

plt.rcParams["ps.fonttype"] = 42
plt.rcParams["pdf.fonttype"] = 42


# ==========================================================
# LOAD DATA
# ==========================================================

all_data = []

for country, filename in COUNTRY_FILES.items():
    path = DATA_DIR / filename

    if not path.exists():
        print(f"File not found: {path}")
        continue

    df = pd.read_csv(path)
    df["country"] = country
    df["r"] = RATES[country]
    df["rho_period"] = df["r"] * df["phi_period"]

    all_data.append(df)

if len(all_data) == 0:
    raise FileNotFoundError("No files were found.")

data = pd.concat(all_data, ignore_index=True)

print("Loaded countries:", sorted(data["country"].unique()))


# ==========================================================
# 1. Average rho over all periods: all countries
# ==========================================================

plt.figure(figsize=(11, 6))

for country in COUNTRY_FILES.keys():
    df_country = data[data["country"] == country]

    if df_country.empty:
        continue

    df_mean = (
        df_country
        .groupby("gamma", as_index=False)["rho_period"]
        .mean()
        .sort_values("gamma")
    )

    plt.plot(
        df_mean["gamma"],
        df_mean["rho_period"],
        linewidth=2.5,
        color=COUNTRY_COLORS.get(country),
        label=country
    )

plt.xlabel(r"$\gamma$")
plt.ylabel(r"$\rho(\gamma)=r\varphi(\gamma)$")
plt.grid(alpha=0.3)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()


# ==========================================================
# 2. All countries by period
# ==========================================================

for period in PERIODS:
    df_period = data[data["period"] == period]

    if df_period.empty:
        continue

    plt.figure(figsize=(11, 6))

    for country in COUNTRY_FILES.keys():
        df_country = (
            df_period[df_period["country"] == country]
            .sort_values("gamma")
        )

        if df_country.empty:
            continue

        plt.plot(
            df_country["gamma"],
            df_country["rho_period"],
            linewidth=2.3,
            color=COUNTRY_COLORS.get(country),
            label=country
        )

    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$\rho(\gamma)=r\varphi(\gamma)$")
    plt.title(f"Discount coefficient by countries, period {period}")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()


# ==========================================================
# 3. One country over periods
# ==========================================================

for country in COUNTRY_FILES.keys():
    df_country = data[data["country"] == country]

    if df_country.empty:
        continue

    plt.figure(figsize=(10, 6))

    for period in PERIODS:
        df_period = (
            df_country[df_country["period"] == period]
            .sort_values("gamma")
        )

        if df_period.empty:
            continue

        plt.plot(
            df_period["gamma"],
            df_period["rho_period"],
            linewidth=2.5,
            linestyle=PERIOD_STYLES.get(period, "-"),
            label=period
        )

    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$\rho(\gamma)=r\varphi(\gamma)$")
    plt.title(f"{country}: discount coefficient over periods")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================================
# 4. Save rho data
# ==========================================================

data.to_csv("rho_periods_all_countries.csv", index=False)

print("Saved: rho_periods_all_countries.csv")