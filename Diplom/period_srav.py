import pandas as pd
import numpy as np
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

COUNTRY_GROUPS = {
    "Developed countries": ["Germany", "Sweden", "Netherlands", "Australia", "USA", "Italy", "France", "Japan"],
    "Developing and transition economies": ["Russia", "China", "Kazahstan"],
    "European countries": ["Germany", "France", "Netherlands", "Sweden", "UK", "Italy", "Spain", "Poland"],
    "Asian countries": ["China", "Japan", "Kazahstan", "Russia"],
    "USA and Australia": ["USA", "Australia"],
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
    all_data.append(df)

if len(all_data) == 0:
    raise FileNotFoundError("No period files were found.")

data = pd.concat(all_data, ignore_index=True)

print("Loaded countries:", sorted(data["country"].unique()))
print("Loaded periods:", sorted(data["period"].unique()))


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================



def setup_plot(title=None):
    if title is not None:
        plt.title(title)

    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$\varphi(\gamma)$")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()


# ==========================================================
# 1. AVERAGE phi OVER ALL PERIODS: COUNTRY COMPARISON
# ==========================================================

plt.figure(figsize=(11, 6))

for country in COUNTRY_FILES.keys():
    df_country = data[data["country"] == country]

    if df_country.empty:
        continue

    df_mean = (
        df_country
        .groupby("gamma", as_index=False)["phi_period"]
        .mean()
        .sort_values("gamma")
    )

    plt.plot(
        df_mean["gamma"],
        df_mean["phi_period"],
        linewidth=2.5,
        color=COUNTRY_COLORS.get(country),
        label=country
    )
setup_plot()


# ==========================================================
# 2. ONE FIGURE FOR EACH PERIOD: ALL COUNTRIES
# ==========================================================

for period in PERIODS:
    df_period = data[data["period"] == period].copy()

    if df_period.empty:
        print(f"No data for period {period}")
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
            df_country["phi_period"],
            linewidth=2.3,
            color=COUNTRY_COLORS.get(country),
            label=country
        )

    setup_plot()


# ==========================================================
# 3. GROUP COMPARISONS OVER ALL PERIODS
# ==========================================================

for group_name, countries in COUNTRY_GROUPS.items():
    plt.figure(figsize=(11, 6))

    plotted = False

    for country in countries:
        df_country = data[data["country"] == country]

        if df_country.empty:
            continue

        df_mean = (
            df_country
            .groupby("gamma", as_index=False)["phi_period"]
            .mean()
            .sort_values("gamma")
        )

        plt.plot(
            df_mean["gamma"],
            df_mean["phi_period"],
            linewidth=2.8,
            color=COUNTRY_COLORS.get(country),
            label=country
        )

        plotted = True

    if not plotted:
        plt.close()
        continue
    setup_plot()


# ==========================================================
# 4. GROUP COMPARISONS BY PERIOD
# ==========================================================

for group_name, countries in COUNTRY_GROUPS.items():
    for period in PERIODS:
        df_period = data[data["period"] == period].copy()

        if df_period.empty:
            continue

        plt.figure(figsize=(11, 6))

        plotted = False

        for country in countries:
            df_country = (
                df_period[df_period["country"] == country]
                .sort_values("gamma")
            )

            if df_country.empty:
                continue

            plt.plot(
                df_country["gamma"],
                df_country["phi_period"],
                linewidth=2.7,
                color=COUNTRY_COLORS.get(country),
                label=country
            )

            plotted = True

        if not plotted:
            plt.close()
            continue
        setup_plot()


# ==========================================================
# 5. ONE COUNTRY OVER PERIODS
# ==========================================================

for country in COUNTRY_FILES.keys():
    df_country_all = data[data["country"] == country].copy()

    if df_country_all.empty:
        continue

    plt.figure(figsize=(10, 6))

    for period in PERIODS:
        df_period = (
            df_country_all[df_country_all["period"] == period]
            .sort_values("gamma")
        )

        if df_period.empty:
            continue

        plt.plot(
            df_period["gamma"],
            df_period["phi_period"],
            linewidth=2.7,
            linestyle=PERIOD_STYLES.get(period, "-"),
            label=period
        )
    setup_plot()