import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from matplotlib.colors import LinearSegmentedColormap


# ==========================================================
# SETTINGS
# ==========================================================

FILE_PATH = "/Users/ulianaalekseeva/PyCharmMiscProject/netherlands_income_merged_filled.csv"
COUNTRY_NAME = "Netherlands"

START_YEAR = 1980
END_YEAR = 2024

DECILES = [
    "p0p10", "p10p20", "p20p30", "p30p40", "p40p50",
    "p50p60", "p60p70", "p70p80", "p80p90", "p90p100"
]

DX = 0.1
DT = 1.0

R_RATE = 0.015
BETA_LIST = np.arange(0.5, 0.95, 0.01)

EPS_PHI = 1e-8

LAMBDA_REG = 1e-6
LAMBDA_SMOOTH = 1e-5

SMOOTH_WINDOW = 31
SMOOTH_POLYORDER = 2

PERIODS = [
    (1980, 1990),
    (1990, 2000),
    (2000, 2010),
    (2010, 2020),
    (2020, 2024)
]

plt.rcParams["ps.fonttype"] = 42
plt.rcParams["pdf.fonttype"] = 42


# ==========================================================
# DATA LOADING
# ==========================================================

df = pd.read_csv(FILE_PATH)

df["tax_type"] = df["tax_type"].astype(str).str.strip().str.lower()
df["decile"] = df["decile"].astype(str).str.strip().str.lower()
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")

df = df.dropna(subset=["tax_type", "decile", "year", "value"]).copy()
df["year"] = df["year"].astype(int)
df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()


# ==========================================================
# BASIC FUNCTIONS
# ==========================================================

def get_year_values(dataframe, tax_type, year):
    d = dataframe[
        (dataframe["tax_type"] == tax_type) &
        (dataframe["year"] == year)
    ].copy()

    d["decile"] = pd.Categorical(d["decile"], categories=DECILES, ordered=True)
    d = d.sort_values("decile")

    if len(d) != 10:
        return None

    values = d["value"].values.astype(float)

    if np.any(~np.isfinite(values)) or np.any(values <= 0) or values.sum() <= 0:
        return None

    return values


def shares_from_values(values):
    return values / values.sum()


def lorenz_from_shares(shares):
    x = np.linspace(0.0, 1.0, 11)
    y = np.concatenate([[0.0], np.cumsum(shares)])
    return x, y


def gamma_from_shares(shares):
    return shares / DX


def smooth_lorenz_curve(shares, n_points=300):
    x_nodes, y_nodes = lorenz_from_shares(shares)
    spline = PchipInterpolator(x_nodes, y_nodes)

    x_dense = np.linspace(0.0, 1.0, n_points)
    y_dense = spline(x_dense)

    y_dense = np.clip(y_dense, 0.0, 1.0)
    y_dense = np.maximum.accumulate(y_dense)
    y_dense[0] = 0.0
    y_dense[-1] = 1.0

    return x_dense, y_dense


def pava_increasing(y, w=None):
    y = np.asarray(y, dtype=float)

    if w is None:
        w = np.ones(len(y), dtype=float)
    else:
        w = np.asarray(w, dtype=float)

    blocks = []

    for i in range(len(y)):
        blocks.append([i, i, y[i], w[i]])

        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            b2 = blocks.pop()
            b1 = blocks.pop()

            total_w = b1[3] + b2[3]
            avg = (b1[2] * b1[3] + b2[2] * b2[3]) / total_w

            blocks.append([b1[0], b2[1], avg, total_w])

    out = np.empty(len(y), dtype=float)

    for start, end, avg, _ in blocks:
        out[start:end + 1] = avg

    return out


def project_to_lorenz(y_interior):
    y_full = np.concatenate([[0.0], y_interior])

    shares = np.diff(y_full)
    shares = np.maximum(shares, 1e-12)
    shares = shares / shares.sum()

    y_proj = np.concatenate([[0.0], np.cumsum(shares)])
    y_proj[-1] = 1.0

    return y_proj


def safe_savgol(y):
    y = np.asarray(y, dtype=float)

    window = min(SMOOTH_WINDOW, len(y) - 1)

    if window % 2 == 0:
        window -= 1

    if window <= SMOOTH_POLYORDER:
        window = SMOOTH_POLYORDER + 3
        if window % 2 == 0:
            window += 1

    if window >= len(y):
        window = len(y) - 1
        if window % 2 == 0:
            window -= 1

    if window <= SMOOTH_POLYORDER or window < 3:
        return y.copy()

    return savgol_filter(y, window_length=window, polyorder=SMOOTH_POLYORDER)


# ==========================================================
# YEARS WITH DATA
# ==========================================================

years_list = []

for year in range(START_YEAR, END_YEAR + 1):
    values_pre = get_year_values(df, "pre", year)
    values_post = get_year_values(df, "post", year)

    if values_pre is not None and values_post is not None:
        years_list.append(year)

print("Years with both pre and post data:")
print(years_list)


# ==========================================================
# PLOT PRE-TAX LORENZ CURVES
# ==========================================================

plt.figure(figsize=(8, 6))

cmap = LinearSegmentedColormap.from_list("blue_red", ["#4aa3ff", "#ff4a4a"])
norm = plt.Normalize(min(years_list), max(years_list))

for year in years_list:
    values_pre = get_year_values(df, "pre", year)
    shares_pre = shares_from_values(values_pre)

    x_smooth, y_smooth = smooth_lorenz_curve(shares_pre)

    plt.plot(
        x_smooth,
        y_smooth,
        color=cmap(norm(year)),
        linewidth=1.8,
        alpha=0.9
    )

plt.plot([0, 1], [0, 1], "--", linewidth=1.5, color="black")
plt.plot([], [], color=cmap(norm(START_YEAR)), label=str(START_YEAR))
plt.plot([], [], color=cmap(norm(END_YEAR)), label=str(END_YEAR))

plt.title(f"Lorenz curves, pre-tax ({COUNTRY_NAME}, {START_YEAR}–{END_YEAR})")
plt.xlabel(r"$x$")
plt.ylabel(r"$y(x,t)$")
plt.grid(alpha=0.3)
plt.legend(title="Year")
plt.tight_layout()
plt.show()


# ==========================================================
# CONSTRUCTION OF psi_0 FROM pre -> post
# ==========================================================

gamma_pre_all = []
gamma_post_all = []

for year in years_list:
    values_pre = get_year_values(df, "pre", year)
    values_post = get_year_values(df, "post", year)

    shares_pre = shares_from_values(values_pre)
    shares_post = shares_from_values(values_post)

    gamma_pre_all.extend(gamma_from_shares(shares_pre))
    gamma_post_all.extend(gamma_from_shares(shares_post))

gamma_pre_all = np.array(gamma_pre_all, dtype=float)
gamma_post_all = np.array(gamma_post_all, dtype=float)

mask = np.isfinite(gamma_pre_all) & np.isfinite(gamma_post_all)
gamma_pre_all = gamma_pre_all[mask]
gamma_post_all = gamma_post_all[mask]

order = np.argsort(gamma_pre_all)
gamma_pre_all = gamma_pre_all[order]
gamma_post_all = gamma_post_all[order]

unique_gamma = np.unique(gamma_pre_all)

unique_psi = []
unique_weights = []

for g in unique_gamma:
    mask_g = np.isclose(gamma_pre_all, g)
    vals = gamma_post_all[mask_g]

    unique_psi.append(np.mean(vals))
    unique_weights.append(len(vals))

unique_psi = np.array(unique_psi, dtype=float)
unique_weights = np.array(unique_weights, dtype=float)

psi_pava = pava_increasing(unique_psi, unique_weights)
psi_mono = np.maximum.accumulate(psi_pava)


def psi0(x):
    x = np.asarray(x, dtype=float)
    return np.interp(x, unique_gamma, psi_mono)


gamma_dense = np.linspace(unique_gamma.min(), unique_gamma.max(), 1000)
psi_dense = psi0(gamma_dense)

plt.figure(figsize=(8, 6))

plt.scatter(
    gamma_pre_all,
    gamma_post_all,
    s=18,
    alpha=0.18,
    label="Data points"
)

plt.plot(
    gamma_dense,
    psi_dense,
    linewidth=3,
    label=r"$\psi_0(\gamma)$"
)

plt.plot(
    gamma_dense,
    gamma_dense,
    "--",
    linewidth=1.2,
    label=r"$\psi_0(\gamma)=\gamma$"
)

plt.title(rf"Redistribution function $\psi_0(\gamma)$, {COUNTRY_NAME}")
plt.xlabel(r"$\gamma^{pre}$")
plt.ylabel(r"$\gamma^{post}$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# PRE-TAX CURVES BY YEAR
# ==========================================================

pre_curves = {}

for year in years_list:
    values_pre = get_year_values(df, "pre", year)
    shares_pre = shares_from_values(values_pre)

    x_pre, y_pre = lorenz_from_shares(shares_pre)
    gamma_pre = gamma_from_shares(shares_pre)

    pre_curves[year] = {
        "x": x_pre,
        "y": y_pre,
        "gamma": gamma_pre
    }

all_transition_years = []

for year in sorted(pre_curves.keys()):
    if year + 1 in pre_curves:
        all_transition_years.append((year, year + 1))

print("Number of transitions:", len(all_transition_years))
print(all_transition_years[:10], "...")


# ==========================================================
# MAIN FUNCTION: RUN MODEL FOR GIVEN beta
# ==========================================================

def run_for_beta(beta_value):
    lambda_coef = R_RATE * DT / (1.0 - beta_value)

    def one_step_rhs_from_phi_nodes(y0, gamma0, phi_nodes):
        psi_vals = psi0(gamma0)
        q_vals = (1.0 - phi_nodes) * psi_vals

        A_vals = DX * np.cumsum(q_vals)
        B_val = DX * np.sum(q_vals)

        rhs = lambda_coef * (A_vals - y0[1:] * B_val)
        return rhs

    def initial_phi_for_year(gamma0):
        order = np.argsort(gamma0)
        phi0_sorted = np.linspace(beta_value + 0.5, beta_value + 0.02, 10)

        phi0 = np.empty(10)
        phi0[order] = phi0_sorted

        return phi0

    def make_constraints_for_year(gamma0):
        constraints = []

        for j in range(10):
            constraints.append({
                "type": "ineq",
                "fun": lambda x, j=j: x[j] - (beta_value + EPS_PHI)
            })

        order = np.argsort(gamma0)

        for k in range(9):
            j_left = order[k]
            j_right = order[k + 1]

            constraints.append({
                "type": "ineq",
                "fun": lambda x, j_left=j_left, j_right=j_right:
                    x[j_left] - x[j_right]
            })

        return constraints

    phi_by_year = {}
    year_results = []

    for year0, year1 in all_transition_years:
        y0 = pre_curves[year0]["y"]
        y1 = pre_curves[year1]["y"]
        gamma0 = pre_curves[year0]["gamma"]

        constraints = make_constraints_for_year(gamma0)

        def objective(phi_nodes):
            rhs = one_step_rhs_from_phi_nodes(y0, gamma0, phi_nodes)

            y_model_next = y0[1:] + rhs
            y_obs_next = y1[1:]

            sse = np.sum((y_model_next - y_obs_next) ** 2)

            order_g = np.argsort(gamma0)
            phi_sorted = phi_nodes[order_g]

            reg_l2 = LAMBDA_REG * np.sum((phi_nodes - beta_value) ** 2)
            reg_smooth = LAMBDA_SMOOTH * np.sum(np.diff(phi_sorted, n=2) ** 2)

            return sse + reg_l2 + reg_smooth

        res = minimize(
            objective,
            initial_phi_for_year(gamma0),
            method="SLSQP",
            constraints=constraints,
            options={
                "maxiter": 3000,
                "ftol": 1e-12,
                "disp": False
            }
        )

        phi_nodes = res.x

        rhs = one_step_rhs_from_phi_nodes(y0, gamma0, phi_nodes)
        y_model_next = y0[1:] + rhs
        J_one = np.sum((y_model_next - y1[1:]) ** 2)

        order_g = np.argsort(gamma0)
        phi_sorted = phi_nodes[order_g]

        phi_by_year[(year0, year1)] = {
            "gamma_nodes": gamma0.copy(),
            "phi_nodes": phi_nodes.copy(),
            "success": res.success,
            "J_one": J_one
        }

        year_results.append({
            "year0": year0,
            "year1": year1,
            "success": res.success,
            "J_one": J_one,
            "min_phi": float(np.min(phi_nodes)),
            "max_phi": float(np.max(phi_nodes)),
            "decreasing": bool(np.all(np.diff(phi_sorted) <= 1e-10))
        })

    common_gamma = np.linspace(
        min(np.min(item["gamma_nodes"]) for item in phi_by_year.values()),
        max(np.max(item["gamma_nodes"]) for item in phi_by_year.values()),
        300
    )

    all_phi_smooth = []
    labels = []

    for (year0, year1), item in phi_by_year.items():
        gamma_nodes = item["gamma_nodes"]
        phi_nodes = item["phi_nodes"]

        order = np.argsort(gamma_nodes)
        g = gamma_nodes[order]
        p = phi_nodes[order]

        g_unique, idx = np.unique(g, return_index=True)
        p_unique = p[idx]

        if len(g_unique) >= 3:
            spline = PchipInterpolator(g_unique, p_unique, extrapolate=True)
            phi_common = spline(common_gamma)
        else:
            phi_common = np.interp(common_gamma, g_unique, p_unique)

        phi_common = np.maximum(phi_common, beta_value + EPS_PHI)

        phi_smooth = safe_savgol(phi_common)

        phi_smooth = np.maximum(phi_smooth, beta_value + EPS_PHI)
        phi_smooth = np.minimum.accumulate(phi_smooth)

        all_phi_smooth.append(phi_smooth)
        labels.append(f"{year0}->{year1}")

    all_phi_smooth = np.array(all_phi_smooth)

    phi_star_vals = np.mean(all_phi_smooth, axis=0)
    phi_star_vals = np.maximum(phi_star_vals, beta_value + EPS_PHI)
    phi_star_vals = np.minimum.accumulate(phi_star_vals)

    def phi_star_raw(gamma):
        gamma = np.asarray(gamma, dtype=float)
        return np.interp(gamma, common_gamma, phi_star_vals)

    def one_step_with_phi(y_now, phi_function):
        shares_now = np.diff(y_now)
        gamma_now = shares_now / DX

        phi_vals = phi_function(gamma_now)
        phi_vals = np.maximum(phi_vals, beta_value + EPS_PHI)

        psi_vals = psi0(gamma_now)

        q_vals = (1.0 - phi_vals) * psi_vals

        A_vals = DX * np.cumsum(q_vals)
        B_val = DX * np.sum(q_vals)

        rhs = lambda_coef * (A_vals - y_now[1:] * B_val)

        y_next = y_now[1:] + rhs
        return project_to_lorenz(y_next)

    model_years = sorted(pre_curves.keys())

    y_model = {
        model_years[0]: pre_curves[model_years[0]]["y"].copy()
    }

    for year in model_years[:-1]:
        next_year = year + 1

        if next_year in pre_curves:
            y_model[next_year] = one_step_with_phi(
                y_model[year],
                phi_star_raw
            )

    J_traj = 0.0
    sum_y_obs = 0.0

    for year in sorted(y_model.keys()):
        y_mod = y_model[year]
        y_obs = pre_curves[year]["y"]

        J_traj += np.sum((y_mod[1:] - y_obs[1:]) ** 2)
        sum_y_obs += np.sum(y_obs[1:])

    Q_norm = np.sqrt(J_traj) / sum_y_obs

    period_phi = {}

    for start, end in PERIODS:
        phi_list = []

        for label, phi_i in zip(labels, all_phi_smooth):
            year0 = int(label.split("->")[0])

            if start <= year0 < end:
                phi_list.append(phi_i)

        if len(phi_list) > 0:
            phi_period = np.mean(phi_list, axis=0)
            phi_period = np.maximum(phi_period, beta_value + EPS_PHI)
            phi_period = np.minimum.accumulate(phi_period)

            period_phi[(start, end)] = phi_period

    period_rho = {
        period: R_RATE * phi_vals
        for period, phi_vals in period_phi.items()
    }

    return {
        "beta": beta_value,
        "J": J_traj,
        "Q": Q_norm,
        "sum_y_obs": sum_y_obs,
        "common_gamma": common_gamma,
        "phi_star": phi_star_vals,
        "phi_star_raw_func": phi_star_raw,
        "period_phi": period_phi,
        "period_rho": period_rho,
        "all_phi_smooth": all_phi_smooth,
        "labels": labels,
        "y_model": y_model,
        "pre_curves": pre_curves,
        "year_results": pd.DataFrame(year_results)
    }


# ==========================================================
# BETA SELECTION
# ==========================================================

all_beta_results = []

for beta in BETA_LIST:
    print("\nRunning beta =", beta)

    result = run_for_beta(beta)

    all_beta_results.append({
        "beta": beta,
        "J": result["J"],
        "Q": result["Q"],
        "sum_y_obs": result["sum_y_obs"],
        "result": result
    })

beta_results_df = pd.DataFrame([
    {
        "beta": item["beta"],
        "J": item["J"],
        "Q": item["Q"],
        "sum_y_obs": item["sum_y_obs"]
    }
    for item in all_beta_results
])

print("\n==============================")
print("BETA SELECTION RESULTS")
print("==============================")
print(beta_results_df)

best_item = min(all_beta_results, key=lambda x: x["Q"])
best_beta = best_item["beta"]
best_result = best_item["result"]

print("\n==============================")
print("BEST BETA")
print("==============================")
print("Best beta:", best_beta)
print("Best J:", best_item["J"])
print("Best Q = sqrt(J)/sum(y_obs):", best_item["Q"])


# ==========================================================
# PLOT Q(beta)
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    beta_results_df["beta"],
    beta_results_df["Q"],
    marker="o",
    linewidth=2
)

plt.title(rf"Selection of $\beta$ by normalized criterion, {COUNTRY_NAME}")
plt.xlabel(r"$\beta$")
plt.ylabel(r"$\sqrt{J}/\sum y_j^{obs}$")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================================================
# PLOT phi* FOR BEST BETA
# ==========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    best_result["common_gamma"],
    best_result["phi_star"],
    color="black",
    linewidth=4,
    label=rf"$\varphi^*(\gamma)$, $\beta={best_beta}$"
)

plt.axhline(
    best_beta,
    linestyle="--",
    color="black",
    linewidth=2,
    alpha=0.8,
    label=r"$\beta$"
)

plt.title(rf"Recovered $\varphi^*(\gamma)$, {COUNTRY_NAME}")
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$\varphi(\gamma)$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# PLOT rho(gamma)=r*phi(gamma) BY PERIODS
# ==========================================================

plt.figure(figsize=(10, 6))

for (start, end), rho_vals in best_result["period_rho"].items():
    plt.plot(
        best_result["common_gamma"],
        rho_vals,
        linewidth=3,
        label=f"{start}–{end}"
    )

plt.title(rf"Discount rate $\rho(\gamma)=r\varphi(\gamma)$ by periods, {COUNTRY_NAME}")
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$\rho(\gamma)$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# PLOT phi(gamma) BY PERIODS FOR BEST BETA
# ==========================================================

plt.figure(figsize=(10, 6))

for (start, end), phi_vals in best_result["period_phi"].items():
    plt.plot(
        best_result["common_gamma"],
        phi_vals,
        linewidth=3,
        label=f"{start}–{end}"
    )

plt.axhline(
    best_beta,
    linestyle="--",
    color="black",
    linewidth=2,
    alpha=0.8,
    label=r"$\beta$"
)

plt.title(rf"Functions $\varphi(\gamma)$ by periods, {COUNTRY_NAME}")
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$\varphi(\gamma)$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# OBSERVED VS MODEL LORENZ CURVES FOR BEST BETA
# ==========================================================

years_to_plot = [1980, 1990, 2000, 2010, 2020]

plt.figure(figsize=(9, 6))

for year in years_to_plot:
    if year in best_result["y_model"]:

        x = pre_curves[year]["x"]
        y_obs = pre_curves[year]["y"]
        y_mod = best_result["y_model"][year]

        plt.plot(x, y_obs, linewidth=2, label=f"observed {year}")
        plt.plot(x, y_mod, "--", linewidth=2, label=f"model {year}")

plt.plot([0, 1], [0, 1], "--", color="black")

plt.title(rf"Observed vs model Lorenz curves, {COUNTRY_NAME}, $\beta={best_beta}$")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(alpha=0.3)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()


# ==========================================================
# SAVE RESULTS
# ==========================================================

beta_results_df.to_csv(
    f"beta_selection_{COUNTRY_NAME.lower()}.csv",
    index=False
)

phi_save_df = pd.DataFrame({
    "country": COUNTRY_NAME,
    "beta": best_beta,
    "gamma": best_result["common_gamma"],
    "phi_star": best_result["phi_star"],
    "rho_star": R_RATE * best_result["phi_star"]
})

phi_save_df.to_csv(
    f"phi_rho_star_{COUNTRY_NAME.lower()}.csv",
    index=False
)

period_rows = []

for (start, end), phi_vals in best_result["period_phi"].items():
    rho_vals = R_RATE * phi_vals

    for gamma_val, phi_val, rho_val in zip(
        best_result["common_gamma"],
        phi_vals,
        rho_vals
    ):
        period_rows.append({
            "country": COUNTRY_NAME,
            "beta": best_beta,
            "period_start": start,
            "period_end": end,
            "period": f"{start}-{end}",
            "gamma": gamma_val,
            "phi_period": phi_val,
            "rho_period": rho_val
        })

period_df = pd.DataFrame(period_rows)

period_df.to_csv(
    f"phi_rho_periods_{COUNTRY_NAME.lower()}.csv",
    index=False
)

print("\nSaved files:")
print(f"beta_selection_{COUNTRY_NAME.lower()}.csv")
print(f"phi_rho_star_{COUNTRY_NAME.lower()}.csv")
print(f"phi_rho_periods_{COUNTRY_NAME.lower()}.csv")

# Save psi_0 function

psi_save_df = pd.DataFrame({
    "country": COUNTRY_NAME,
    "gamma_pre": gamma_dense,
    "psi_0": psi_dense
})

psi_save_df.to_csv(
    f"psi0_{COUNTRY_NAME.lower()}.csv",
    index=False
)

print(f"psi0_{COUNTRY_NAME.lower()}.csv")