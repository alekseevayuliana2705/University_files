import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# SETTINGS
# ==========================================================

BASE_FILE_PATH = "/Users/ulianaalekseeva/PyCharmMiscProject/russia_income_merged_filled.csv"
BASE_COUNTRY = "Russia"
BASE_YEAR = 2024
BASE_TAX_TYPE = "pre"

PSI_FOLDER = "/Users/ulianaalekseeva/PyCharmMiscProject/"

PSI_FILES = {
    "Russia": "psi0_russia.csv",
    "Germany": "psi0_germany.csv",
    "USA": "psi0_usa.csv",
    "UK": "psi0_uk.csv",
    "Sweden": "psi0_sweden.csv",
    "Spain": "psi0_spain.csv",
    "Poland": "psi0_poland.csv",
    "Netherlands": "psi0_netherlands.csv",
    "France": "psi0_france.csv",
    "China": "psi0_china.csv",
    "Japan": "psi0_japan.csv",
    "Italy": "psi0_italy.csv",
    "Australia": "psi0_australia.csv",
    "Kazahstan": "psi0_kazahstan.csv"
}

DECILES = [
    "p0p10", "p10p20", "p20p30", "p30p40", "p40p50",
    "p50p60", "p60p70", "p70p80", "p80p90", "p90p100"
]

DX = 0.1


# ==========================================================
# FUNCTIONS
# ==========================================================

def get_year_values(file_path, tax_type, year):
    df = pd.read_csv(file_path)

    df["tax_type"] = df["tax_type"].astype(str).str.strip().str.lower()
    df["decile"] = df["decile"].astype(str).str.strip().str.lower()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["tax_type", "decile", "year", "value"]).copy()
    df["year"] = df["year"].astype(int)

    d = df[
        (df["tax_type"] == tax_type.lower()) &
        (df["year"] == year)
    ].copy()

    d["decile"] = pd.Categorical(d["decile"], categories=DECILES, ordered=True)
    d = d.sort_values("decile")

    if len(d) != 10:
        raise ValueError(f"Expected 10 deciles for {year}, got {len(d)}")

    values = d["value"].values.astype(float)

    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Invalid income values")

    return values


def shares_from_values(values):
    values = np.asarray(values, dtype=float)
    return values / values.sum()


def gamma_from_shares(shares):
    return shares / DX


def shares_from_gamma(gamma):
    shares = gamma * DX
    shares = np.maximum(shares, 1e-12)
    shares = shares / shares.sum()
    return shares


def lorenz_from_shares(shares):
    x = np.linspace(0, 1, len(shares) + 1)
    y = np.concatenate([[0], np.cumsum(shares)])
    y[-1] = 1.0
    return x, y


def gini_from_lorenz(x, y):
    area = np.trapz(y, x)
    return 1 - 2 * area


def load_psi_function(file_path):
    df = pd.read_csv(file_path)

    gamma_col = "gamma_pre" if "gamma_pre" in df.columns else "gamma"
    psi_col = "psi_0" if "psi_0" in df.columns else "psi0"

    gamma = df[gamma_col].values.astype(float)
    psi = df[psi_col].values.astype(float)

    order = np.argsort(gamma)
    gamma = gamma[order]
    psi = psi[order]

    def psi_func(x):
        x = np.asarray(x, dtype=float)
        return np.interp(x, gamma, psi)

    return psi_func


def apply_tax_system(base_shares, psi_func):
    gamma_base = gamma_from_shares(base_shares)
    gamma_new = psi_func(gamma_base)

    shares_new = shares_from_gamma(gamma_new)

    return shares_new, gamma_new


# ==========================================================
# BASE DISTRIBUTION
# ==========================================================

base_values = get_year_values(
    BASE_FILE_PATH,
    BASE_TAX_TYPE,
    BASE_YEAR
)

base_shares = shares_from_values(base_values)
base_gamma = gamma_from_shares(base_shares)

x_base, y_base = lorenz_from_shares(base_shares)
gini_base = gini_from_lorenz(x_base, y_base)

print("\n==============================")
print("BASE DISTRIBUTION")
print("==============================")
print("Country:", BASE_COUNTRY)
print("Year:", BASE_YEAR)
print("Tax type:", BASE_TAX_TYPE)
print("Base Gini:", round(gini_base, 4))


# ==========================================================
# APPLY DIFFERENT TAX SYSTEMS
# ==========================================================

results = []

curves = {
    "Base": {
        "x": x_base,
        "y": y_base,
        "gini": gini_base,
        "shares": base_shares
    }
}

for country, filename in PSI_FILES.items():
    psi_path = os.path.join(PSI_FOLDER, filename)

    if not os.path.exists(psi_path):
        print(f"Skipped {country}: file not found {psi_path}")
        continue

    psi_func = load_psi_function(psi_path)

    new_shares, new_gamma = apply_tax_system(base_shares, psi_func)

    x_new, y_new = lorenz_from_shares(new_shares)
    gini_new = gini_from_lorenz(x_new, y_new)

    delta_gini = gini_base - gini_new
    relative_reduction = delta_gini / gini_base

    results.append({
        "tax_system": country,
        "base_country": BASE_COUNTRY,
        "base_year": BASE_YEAR,
        "gini_before": gini_base,
        "gini_after": gini_new,
        "delta_gini": delta_gini,
        "relative_reduction": relative_reduction
    })

    curves[country] = {
        "x": x_new,
        "y": y_new,
        "gini": gini_new,
        "shares": new_shares,
        "gamma": new_gamma
    }


results_df = pd.DataFrame(results)

# Сортируем по итоговому индексу Джини:
# чем меньше gini_after, тем сильнее итоговое выравнивание
results_df = results_df.sort_values("gini_after", ascending=True)

print("\n==============================")
print("COMPARISON OF TAX SYSTEMS")
print("==============================")
print(results_df)


# ==========================================================
# PLOT LORENZ CURVES
# ==========================================================

plt.figure(figsize=(10, 7))

plt.plot(
    x_base,
    y_base,
    color="black",
    linewidth=4,
    label=f"Base: {BASE_COUNTRY} {BASE_YEAR}, Gini={gini_base:.3f}"
)

for country, item in curves.items():
    if country == "Base":
        continue

    plt.plot(
        item["x"],
        item["y"],
        linewidth=2,
        label=f"{country}, Gini={item['gini']:.3f}"
    )

plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5)

plt.xlabel(r"$x$")
plt.ylabel(r"$y(x)$")
plt.grid(alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()


# ==========================================================
# BARPLOT: GINI AFTER REDISTRIBUTION
# ==========================================================

plt.figure(figsize=(11, 5))

plt.bar(
    results_df["tax_system"],
    results_df["gini_after"]
)

plt.axhline(
    gini_base,
    color="red",
    linestyle="--",
    linewidth=2,
    label=fr"Base Gini = {gini_base:.3f}"
)

plt.xlabel("Country tax system")
plt.ylabel("Gini index after redistribution")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# OPTIONAL: BARPLOT WITH BOTH BEFORE AND AFTER
# ==========================================================

plt.figure(figsize=(11, 5))

plt.bar(
    results_df["tax_system"],
    results_df["gini_after"],
    label=r"$Gini_{after}$"
)

plt.axhline(
    gini_base,
    color="red",
    linestyle="--",
    linewidth=2,
    label=r"$Gini_{before}$"
)

plt.xlabel("Country tax system")
plt.ylabel("Gini index")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# SAVE RESULTS
# ==========================================================

comparison_file = f"tax_system_comparison_base_{BASE_COUNTRY.lower()}_{BASE_YEAR}.csv"

results_df.to_csv(
    comparison_file,
    index=False
)

shares_rows = []

for country, item in curves.items():
    shares = item["shares"]

    for j, share in enumerate(shares, start=1):
        shares_rows.append({
            "base_country": BASE_COUNTRY,
            "base_year": BASE_YEAR,
            "tax_system": country,
            "decile": DECILES[j - 1],
            "income_share": share,
            "gini": item["gini"]
        })

shares_df = pd.DataFrame(shares_rows)

shares_file = f"tax_system_new_shares_base_{BASE_COUNTRY.lower()}_{BASE_YEAR}.csv"

shares_df.to_csv(
    shares_file,
    index=False
)

print("\nSaved:")
print(comparison_file)
print(shares_file)