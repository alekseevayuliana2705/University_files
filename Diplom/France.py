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

FILE_PATH = "/Users/ulianaalekseeva/PyCharmMiscProject/france_income_merged_filled.csv"
COUNTRY_NAME = "France"

START_YEAR = 1980
END_YEAR = 2024

DECILES = [
    "p0p10", "p10p20", "p20p30", "p30p40", "p40p50",
    "p50p60", "p60p70", "p70p80", "p80p90", "p90p100"
]

DX = 0.1
DT = 1.0

R_RATE = 0.0215
BETA_MODEL = 0.73

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
# FUNCTIONS
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


# ==========================================================
# 1. LORENZ CURVES
# ==========================================================

years_list = []

for year in range(START_YEAR, END_YEAR + 1):
    values_pre = get_year_values(df, "pre", year)
    if values_pre is not None:
        years_list.append(year)

plt.figure(figsize=(8, 6))

COLOR_START = "#0057FF"   # ярко-синий для 1980
COLOR_END = "#00B050"     # зелёный для 2024
COLOR_OTHER = "#BDBDBD"   # серый для остальных годов

for year in years_list:
    values_pre = get_year_values(df, "pre", year)
    shares_pre = shares_from_values(values_pre)

    x_smooth, y_smooth = smooth_lorenz_curve(shares_pre)

    if year == START_YEAR:
        color = COLOR_START
        linewidth = 2.8
        alpha = 1.0
        zorder = 3
        label = str(START_YEAR)
    elif year == END_YEAR:
        color = COLOR_END
        linewidth = 2.8
        alpha = 1.0
        zorder = 3
        label = str(END_YEAR)
    else:
        color = COLOR_OTHER
        linewidth = 1.1
        alpha = 0.45
        zorder = 1
        label = None

    plt.plot(
        x_smooth,
        y_smooth,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
        label=label
    )

plt.plot([0, 1], [0, 1], "--", linewidth=1.5, color="black", label="Equality line")

plt.xlabel(r"$x$")
plt.ylabel(r"$y(x,t)$")
plt.grid(alpha=0.3)
plt.legend(title="Year")
plt.tight_layout()
plt.show()
# ==========================================================
# 2. CONSTRUCTION OF psi_0
# ==========================================================

gamma_pre_all = []
gamma_post_all = []

for year in years_list:
    values_pre = get_year_values(df, "pre", year)
    values_post = get_year_values(df, "post", year)

    if values_pre is None or values_post is None:
        continue

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

plt.xlabel(r"$\gamma$")
plt.ylabel(r"$\psi_0(\gamma)$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# 3. PRE-TAX CURVES BY YEAR
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

lambda_coef = R_RATE * DT / (1.0 - BETA_MODEL)


# ==========================================================
# 4. RECOVERY OF phi FOR EACH YEAR
# ==========================================================

def one_step_rhs_from_phi_nodes(y0, gamma0, phi_nodes):
    psi_vals = psi0(gamma0)
    q_vals = (1.0 - phi_nodes) * psi_vals

    A_vals = DX * np.cumsum(q_vals)
    B_val = DX * np.sum(q_vals)

    rhs = lambda_coef * (A_vals - y0[1:] * B_val)

    return rhs


def initial_phi_for_year(gamma0):
    order = np.argsort(gamma0)
    phi0_sorted = np.linspace(BETA_MODEL + 0.5, BETA_MODEL + 0.02, 10)

    phi0 = np.empty(10)
    phi0[order] = phi0_sorted

    return phi0


def make_constraints_for_year(gamma0):
    constraints = []

    # phi_j >= beta
    for j in range(10):
        constraints.append({
            "type": "ineq",
            "fun": lambda x, j=j: x[j] - (BETA_MODEL + EPS_PHI)
        })

    # monotonicity with respect to gamma
    order = np.argsort(gamma0)

    for k in range(9):
        j_left = order[k]
        j_right = order[k + 1]

        constraints.append({
            "type": "ineq",
            "fun": lambda x, j_left=j_left, j_right=j_right: x[j_left] - x[j_right]
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

        reg_l2 = LAMBDA_REG * np.sum((phi_nodes - BETA_MODEL) ** 2)
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
    J_sse = np.sum((y_model_next - y1[1:]) ** 2)

    phi_by_year[(year0, year1)] = {
        "gamma_nodes": gamma0.copy(),
        "phi_nodes": phi_nodes.copy(),
        "success": res.success,
        "J_one": J_sse
    }

    order_g = np.argsort(gamma0)
    phi_sorted = phi_nodes[order_g]

    year_results.append({
        "year0": year0,
        "year1": year1,
        "success": res.success,
        "J_one": J_sse,
        "min_phi": float(np.min(phi_nodes)),
        "max_phi": float(np.max(phi_nodes)),
        "decreasing": bool(np.all(np.diff(phi_sorted) <= 1e-10))
    })

year_results_df = pd.DataFrame(year_results)

print("\n==============================")
print("YEARLY RESULTS")
print("==============================")
print(year_results_df.head(15))
print("\nAverage one-step J by year:", year_results_df["J_one"].mean())


# ==========================================================
# 5. SMOOTHING YEARLY phi ON A COMMON GRID
# ==========================================================

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

    phi_common = np.maximum(phi_common, BETA_MODEL + EPS_PHI)

    window = min(SMOOTH_WINDOW, len(phi_common) - 1)

    if window % 2 == 0:
        window -= 1

    if window <= SMOOTH_POLYORDER:
        window = SMOOTH_POLYORDER + 3
        if window % 2 == 0:
            window += 1

    phi_smooth = savgol_filter(
        phi_common,
        window_length=window,
        polyorder=SMOOTH_POLYORDER
    )

    phi_smooth = np.maximum(phi_smooth, BETA_MODEL + EPS_PHI)
    phi_smooth = np.minimum.accumulate(phi_smooth)

    all_phi_smooth.append(phi_smooth)
    labels.append(f"{year0}->{year1}")

all_phi_smooth = np.array(all_phi_smooth)


# ==========================================================
# 6. AGGREGATED phi* WITHOUT INTERMEDIATE AVERAGING
# ==========================================================

phi_star_vals = np.mean(all_phi_smooth, axis=0)

phi_star_vals = np.maximum(phi_star_vals, BETA_MODEL + EPS_PHI)
phi_star_vals = np.minimum.accumulate(phi_star_vals)


def phi_star_raw(gamma):
    gamma = np.asarray(gamma, dtype=float)
    return np.interp(gamma, common_gamma, phi_star_vals)


# ==========================================================
# 7. YEARLY phi AND AGGREGATED phi*
# ==========================================================

distances = np.mean((all_phi_smooth - phi_star_vals) ** 2, axis=1)
threshold = distances.mean() + 1.5 * distances.std()

outlier_indices = np.where(distances > threshold)[0]

print("\n==============================")
print("YEARS STRONGLY DEVIATING FROM AVERAGE phi*")
print("==============================")

if len(outlier_indices) == 0:
    print("No strongly deviating years found.")
else:
    for idx in outlier_indices:
        print(labels[idx], "distance =", distances[idx])


plt.figure(figsize=(10, 6))

base_cmap = plt.cm.viridis
base_colors = base_cmap(np.linspace(0.0, 1.0, len(all_phi_smooth)))

outlier_cmap = plt.cm.tab10
outlier_colors = outlier_cmap(np.linspace(0.0, 1.0, max(len(outlier_indices), 1)))
outlier_color_map = {
    idx: outlier_colors[k]
    for k, idx in enumerate(outlier_indices)
}

for i, phi_i in enumerate(all_phi_smooth):

    if i in outlier_indices:
        plt.plot(
            common_gamma,
            phi_i,
            color=outlier_color_map[i],
            linewidth=2.0,
            alpha=1.0,
            label=labels[i]
        )
    else:
        plt.plot(
            common_gamma,
            phi_i,
            color=base_colors[i],
            linewidth=1.1,
            alpha=0.25
        )

plt.plot(
    common_gamma,
    phi_star_vals,
    color="black",
    linewidth=4,
    label=r"$\phi^*$"
)

plt.axhline(
    BETA_MODEL,
    linestyle="--",
    color="black",
    linewidth=2,
    alpha=0.8,
    label=r"$\beta$"
)

plt.xlabel(r"$\gamma$")
plt.ylabel(r"$\phi(\gamma)$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# 8. DYNAMICS OF phi BY PERIODS
# ==========================================================

period_phi = {}

for start, end in PERIODS:
    phi_list = []

    for label, phi_i in zip(labels, all_phi_smooth):
        year0 = int(label.split("->")[0])

        if start <= year0 < end:
            phi_list.append(phi_i)

    if len(phi_list) > 0:
        phi_period = np.mean(phi_list, axis=0)

        phi_period = np.maximum(phi_period, BETA_MODEL + EPS_PHI)
        phi_period = np.minimum.accumulate(phi_period)

        period_phi[(start, end)] = phi_period

plt.figure(figsize=(10, 6))

for (start, end), phi_period in period_phi.items():
    plt.plot(
        common_gamma,
        phi_period,
        linewidth=3,
        label=f"{start}–{end}"
    )

plt.axhline(
    BETA_MODEL,
    linestyle="--",
    color="black",
    linewidth=2,
    alpha=0.8,
    label=r"$\beta$"
)

plt.xlabel(r"$\gamma$")
plt.ylabel(r"$\phi(\gamma)$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# 9. EXPLICIT FLEXIBLE APPROXIMATION OF phi*
# ==========================================================

def phi_flexible(gamma, params):
    A1, c1, d1, A2, c2, d2, A3, c3 = params
    gamma = np.asarray(gamma, dtype=float)

    return (
        BETA_MODEL
        + A1 / (1.0 + np.exp(c1 * (gamma - d1)))
        + A2 * gamma / (1.0 + np.exp(c2 * (gamma - d2)))
        + A3 * np.exp(-c3 * gamma)
    )


def approximation_objective(params):
    A1, c1, d1, A2, c2, d2, A3, c3 = params

    if A1 <= 0 or c1 <= 0 or A2 <= 0 or c2 <= 0 or A3 <= 0 or c3 <= 0:
        return 1e12

    phi_fit = phi_flexible(common_gamma, params)

    if np.any(~np.isfinite(phi_fit)):
        return 1e12

    if np.any(phi_fit <= BETA_MODEL):
        return 1e12

    mse = np.mean((phi_fit - phi_star_vals) ** 2)

    # штраф за нарушение убывания
    diff = np.diff(phi_fit)
    monotone_penalty = 1e7 * np.sum(np.maximum(diff, 0.0) ** 2)

    return mse + monotone_penalty


A0 = max(phi_star_vals[0] - BETA_MODEL, 1e-3)

x0 = np.array([
    0.35 * A0, 8.0, 0.5,
    0.15 * A0, 3.0, 1.6,
    0.50 * A0, 0.8
])

bounds = [
    (1e-8, 10.0),                         # A1
    (1e-8, 30.0),                         # c1
    (common_gamma.min(), common_gamma.max()), # d1

    (1e-8, 10.0),                         # A2
    (1e-8, 30.0),                         # c2
    (common_gamma.min(), common_gamma.max()), # d2

    (1e-8, 10.0),                         # A3
    (1e-8, 30.0)                          # c3
]

res_approx = minimize(
    approximation_objective,
    x0=x0,
    method="L-BFGS-B",
    bounds=bounds,
    options={
        "maxiter": 20000,
        "ftol": 1e-15
    }
)

params_approx = res_approx.x
phi_star_approx_vals = phi_flexible(common_gamma, params_approx)

MSE_approx = np.mean((phi_star_approx_vals - phi_star_vals) ** 2)

A1, c1, d1, A2, c2, d2, A3, c3 = params_approx

print("\n==============================")
print("EXPLICIT FLEXIBLE APPROXIMATION OF phi*")
print("==============================")
print("success:", res_approx.success)
print(f"MSE = {MSE_approx:.10f}")
print()
print("Parameters:")
print(f"A1 = {A1:.8f}, c1 = {c1:.8f}, d1 = {d1:.8f}")
print(f"A2 = {A2:.8f}, c2 = {c2:.8f}, d2 = {d2:.8f}")
print(f"A3 = {A3:.8f}, c3 = {c3:.8f}")
print()
print("Explicit formula:")
print(
    f"phi*(gamma) = {BETA_MODEL:.6f}"
    f" + {A1:.8f}/(1 + exp({c1:.8f}*(gamma - {d1:.8f})))"
    f" + {A2:.8f}*gamma/(1 + exp({c2:.8f}*(gamma - {d2:.8f})))"
    f" + {A3:.8f}*exp(-{c3:.8f}*gamma)"
)


def phi_star_approx(gamma):
    gamma = np.asarray(gamma, dtype=float)
    return phi_flexible(gamma, params_approx)


plt.figure(figsize=(10, 6))

plt.plot(
    common_gamma,
    phi_star_vals,
    color="black",
    linewidth=4,
    label=r"Raw $\phi^*$"
)

plt.plot(
    common_gamma,
    phi_star_approx_vals,
    "--",
    color="red",
    linewidth=3,
    label=rf"Explicit approximation, MSE={MSE_approx:.2e}"
)

plt.axhline(
    BETA_MODEL,
    linestyle="--",
    color="black",
    linewidth=2,
    alpha=0.8, 
    label=r"$\beta$"
)

plt.xlabel(r"$\gamma$")
plt.ylabel(r"$\phi(\gamma)$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# 10. ONE-STEP FUNCTION
# ==========================================================

def one_step_with_phi(y_now, phi_function):
    shares_now = np.diff(y_now)
    gamma_now = shares_now / DX

    phi_vals = phi_function(gamma_now)
    phi_vals = np.maximum(phi_vals, BETA_MODEL + EPS_PHI)

    psi_vals = psi0(gamma_now)

    q_vals = (1.0 - phi_vals) * psi_vals

    A_vals = DX * np.cumsum(q_vals)
    B_val = DX * np.sum(q_vals)

    rhs = lambda_coef * (A_vals - y_now[1:] * B_val)

    y_next = y_now[1:] + rhs

    return project_to_lorenz(y_next)


# ==========================================================
# 11. FUNCTIONALS FOR A GIVEN phi
# ==========================================================

def compute_functionals(phi_function):
    J_one = 0.0
    n_one = 0

    for year0, year1 in all_transition_years:
        y0 = pre_curves[year0]["y"]
        y1 = pre_curves[year1]["y"]

        y_model_next = one_step_with_phi(y0, phi_function)

        J_one += np.sum((y_model_next[1:] - y1[1:]) ** 2)
        n_one += len(y1[1:])

    MSE_one = J_one / n_one
    RMSE_one = np.sqrt(MSE_one)

    model_years = sorted(pre_curves.keys())

    y_model = {
        model_years[0]: pre_curves[model_years[0]]["y"].copy()
    }

    for year in model_years[:-1]:
        next_year = year + 1

        if next_year in pre_curves:
            y_model[next_year] = one_step_with_phi(
                y_model[year],
                phi_function
            )

    J_traj = 0.0
    n_traj = 0

    years_error = []
    errors = []

    for year in sorted(y_model.keys()):
        y_model_year = y_model[year]
        y_obs = pre_curves[year]["y"]

        err = np.sum((y_model_year[1:] - y_obs[1:]) ** 2)

        J_traj += err
        n_traj += len(y_obs[1:])

        years_error.append(year)
        errors.append(err / len(y_obs[1:]))

    MSE_traj = J_traj / n_traj
    RMSE_traj = np.sqrt(MSE_traj)

    return {
        "J_one": J_one,
        "MSE_one": MSE_one,
        "RMSE_one": RMSE_one,
        "J_traj": J_traj,
        "MSE_traj": MSE_traj,
        "RMSE_traj": RMSE_traj,
        "y_model": y_model,
        "years_error": years_error,
        "errors": errors
    }


res_raw = compute_functionals(phi_star_raw)
res_approx_func = compute_functionals(phi_star_approx)

print("\n==============================")
print("FUNCTIONALS FOR RAW phi*")
print("==============================")
print(f"One-step J       = {res_raw['J_one']:.6f}")
print(f"One-step RMSE    = {res_raw['RMSE_one']:.6f}")
print(f"Trajectory J     = {res_raw['J_traj']:.6f}")
print(f"Trajectory RMSE  = {res_raw['RMSE_traj']:.6f}")

print("\n==============================")
print("FUNCTIONALS FOR APPROXIMATED phi*")
print("==============================")
print(f"One-step J       = {res_approx_func['J_one']:.6f}")
print(f"One-step RMSE    = {res_approx_func['RMSE_one']:.6f}")
print(f"Trajectory J     = {res_approx_func['J_traj']:.6f}")
print(f"Trajectory RMSE  = {res_approx_func['RMSE_traj']:.6f}")


# ==========================================================
# 12. ERROR OVER TIME
# ==========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    res_raw["years_error"],
    res_raw["errors"],
    marker="o",
    linewidth=2,
    label="without approximation"
)

plt.plot(
    res_approx_func["years_error"],
    res_approx_func["errors"],
    marker="o",
    linewidth=2,
    linestyle="--",
    label="with approximation"
)

plt.xlabel("Year")
plt.ylabel("MSE")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# 13. OBSERVED VS MODEL LORENZ CURVES
# ==========================================================

years_to_plot = [1980, 1990, 2000, 2010, 2020]
y_model_approx = res_approx_func["y_model"]

plt.figure(figsize=(9, 6))

for year in years_to_plot:
    if year in y_model_approx:

        x = pre_curves[year]["x"]
        y_obs = pre_curves[year]["y"]
        y_mod = y_model_approx[year]

        plt.plot(x, y_obs, linewidth=2, label=f"observed {year}")
        plt.plot(x, y_mod, "--", linewidth=2, label=f"model {year}")

plt.plot([0, 1], [0, 1], "--", color="black")

plt.xlabel("x")
plt.ylabel("y")
plt.grid(alpha=0.3)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()


# ==========================================================
# 14. SAVE phi* AND PERIOD FUNCTIONS
# ==========================================================

phi_save_df = pd.DataFrame({
    "country": COUNTRY_NAME,
    "gamma": common_gamma,
    "phi_star_raw": phi_star_vals,
    "phi_star_approx": phi_star_approx_vals,
    "beta": BETA_MODEL
})

phi_save_df.to_csv(
    f"phi_star_raw_{COUNTRY_NAME.lower()}.csv",
    index=False
)

print(f"\nSaved: phi_star_raw_{COUNTRY_NAME.lower()}.csv")


period_rows = []

for (start, end), phi_period in period_phi.items():
    for gamma_val, phi_val in zip(common_gamma, phi_period):
        period_rows.append({
            "country": COUNTRY_NAME,
            "period_start": start,
            "period_end": end,
            "period": f"{start}-{end}",
            "gamma": gamma_val,
            "phi_period": phi_val,
            "beta": BETA_MODEL
        })

period_phi_df = pd.DataFrame(period_rows)

period_phi_df.to_csv(
    f"phi_periods_{COUNTRY_NAME.lower()}.csv",
    index=False
)

print(f"Saved: phi_periods_{COUNTRY_NAME.lower()}.csv")