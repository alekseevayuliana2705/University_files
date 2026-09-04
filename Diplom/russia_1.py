# ==========================================================
# INVERSE PROBLEM FOR INCOME DISTRIBUTION DYNAMICS
# Corrected version: global estimation of phi(gamma)
# ==========================================================
#
# Main corrections relative to the older script:
# 1. R_RATE is explicitly treated as the current nominal key rate.
# 2. phi is constrained to the admissible domain beta < phi <= 1.
# 3. phi(gamma) is estimated globally for the trajectory criterion,
#    not by averaging separate one-year optimums.
# 4. psi_0 is anchored by psi_0(1)=1 and clamped outside observed range.
# 5. The Lorenz projection is retained only as a numerical safeguard,
#    and its size is saved as a diagnostic/penalty.
# 6. No PCHIP extrapolation is used for estimated structural functions.
#
# Input CSV must contain columns:
#   year, tax_type, decile, value
# where tax_type is "pre" or "post",
# decile is p0p10, ..., p90p100,
# value is income/value for the decile.
#
# ==========================================================

import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator
from matplotlib.colors import LinearSegmentedColormap


# ==========================================================
# SETTINGS
# ==========================================================

FILE_PATH = "/Users/ulianaalekseeva/PyCharmMiscProject/russia_income_merged_filled.csv"
OUTPUT_DIR = "model_output_russia_corrected"
COUNTRY_NAME = "Russia"

START_YEAR = 1980
END_YEAR = 2024

DECILES = [
    "p0p10", "p10p20", "p20p30", "p30p40", "p40p50",
    "p50p60", "p60p70", "p70p80", "p80p90", "p90p100"
]

DX = 0.1
DT = 1.0

# Current nominal key rate of the Bank of Russia.
# Fix the date in the text of the thesis/report.
R_RATE = 0.145
R_RATE_LABEL = "current nominal key rate"
R_RATE_DATE = "2026-05-06"

# beta is the exponent in utility c^beta, beta in (0,1).
# The model requires phi(gamma) > beta.
BETA_LIST = np.arange(0.30, 0.95, 0.01)

EPS = 1e-8
PHI_UPPER = 1.0 - EPS

# gamma-grid for the unknown function phi(gamma)
N_GAMMA_GRID = 35

# Regularization weights
LAMBDA_SMOOTH_PHI = 1e-3
LAMBDA_REPAIR = 1e-2
LAMBDA_ENDPOINT = 1e-4

# SLSQP settings
MAXITER = 2000
FTOL = 1e-10

# Periods for additional period-specific estimates
PERIODS = [
    (1980, 1990),
    (1990, 2000),
    (2000, 2010),
    (2010, 2020),
    (2020, 2024),
]

SAVE_FIGURES = True
SHOW_FIGURES = True

plt.rcParams["ps.fonttype"] = 42
plt.rcParams["pdf.fonttype"] = 42


# ==========================================================
# DATA STRUCTURES
# ==========================================================

@dataclass
class ModelData:
    years: list
    pre_curves: dict
    post_curves: dict
    transitions: list


@dataclass
class EstimationResult:
    beta: float
    gamma_grid: np.ndarray
    phi_grid: np.ndarray
    J: float
    Q: float
    repair_mean: float
    repair_max: float
    success: bool
    message: str
    y_model: dict
    diagnostics: pd.DataFrame


# ==========================================================
# BASIC UTILITIES
# ==========================================================

def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_or_show(filename: str) -> None:
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()


def pava_increasing(y, w=None):
    """
    Weighted pool-adjacent-violators algorithm.
    Returns the closest nondecreasing vector in weighted L2 sense.
    """
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


def pava_decreasing(y, w=None):
    return -pava_increasing(-np.asarray(y, dtype=float), w)


def project_to_lorenz_from_interior(y_interior):
    """
    Project model output to an admissible discrete Lorenz curve.
    This is a safeguard, not the model itself.
    """
    y_interior = np.asarray(y_interior, dtype=float)
    y_full_raw = np.concatenate([[0.0], y_interior])
    shares_raw = np.diff(y_full_raw)

    shares = np.maximum(shares_raw, 1e-12)
    shares = pava_increasing(shares)
    shares = np.maximum(shares, 1e-12)
    shares = shares / shares.sum()

    y_proj = np.concatenate([[0.0], np.cumsum(shares)])
    y_proj[-1] = 1.0

    repair_norm = float(np.linalg.norm(y_proj[1:] - y_interior))
    return y_proj, repair_norm


def shares_from_values(values):
    values = np.asarray(values, dtype=float)
    return values / values.sum()


def lorenz_from_shares(shares):
    x = np.linspace(0.0, 1.0, 11)
    y = np.concatenate([[0.0], np.cumsum(shares)])
    y[-1] = 1.0
    return x, y


def gamma_from_shares(shares):
    return np.asarray(shares, dtype=float) / DX


def lorenz_curve_from_values(values):
    shares = shares_from_values(values)
    x, y = lorenz_from_shares(shares)
    gamma = gamma_from_shares(shares)
    return {"x": x, "y": y, "shares": shares, "gamma": gamma}


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


# ==========================================================
# DATA LOADING
# ==========================================================

def load_dataframe(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    required = {"year", "tax_type", "decile", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["tax_type"] = df["tax_type"].astype(str).str.strip().str.lower()
    df["decile"] = df["decile"].astype(str).str.strip().str.lower()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["tax_type", "decile", "year", "value"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()

    return df


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


def build_model_data(df: pd.DataFrame) -> ModelData:
    pre_curves = {}
    post_curves = {}
    years = []

    for year in range(START_YEAR, END_YEAR + 1):
        values_pre = get_year_values(df, "pre", year)
        values_post = get_year_values(df, "post", year)

        if values_pre is not None and values_post is not None:
            years.append(year)
            pre_curves[year] = lorenz_curve_from_values(values_pre)
            post_curves[year] = lorenz_curve_from_values(values_post)

    transitions = [(y, y + 1) for y in years if (y + 1) in pre_curves]

    if len(years) < 2:
        raise ValueError("Not enough years with both pre and post data.")

    if len(transitions) == 0:
        raise ValueError("No adjacent-year transitions found.")

    print("Years with both pre and post data:")
    print(years)
    print("Number of adjacent transitions:", len(transitions))

    return ModelData(
        years=years,
        pre_curves=pre_curves,
        post_curves=post_curves,
        transitions=transitions,
    )


# ==========================================================
# CONSTRUCTION OF psi_0 FROM PRE -> POST
# ==========================================================

def build_psi0(model_data: ModelData):
    """
    Builds monotone psi_0(gamma) using pre-tax and post-tax decile densities.
    The point (1,1) is added as a structural anchor.
    """
    gamma_pre_all = []
    gamma_post_all = []

    for year in model_data.years:
        gamma_pre_all.extend(model_data.pre_curves[year]["gamma"])
        gamma_post_all.extend(model_data.post_curves[year]["gamma"])

    gamma_pre_all = np.asarray(gamma_pre_all, dtype=float)
    gamma_post_all = np.asarray(gamma_post_all, dtype=float)

    mask = (
        np.isfinite(gamma_pre_all) &
        np.isfinite(gamma_post_all) &
        (gamma_pre_all > 0) &
        (gamma_post_all > 0)
    )
    gamma_pre_all = gamma_pre_all[mask]
    gamma_post_all = gamma_post_all[mask]

    # Structural anchor: equal distribution remains equal.
    anchor_weight = max(10, len(gamma_pre_all) // 5)
    gamma_pre_all = np.concatenate([gamma_pre_all, np.full(anchor_weight, 1.0)])
    gamma_post_all = np.concatenate([gamma_post_all, np.full(anchor_weight, 1.0)])

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

    unique_psi = np.asarray(unique_psi, dtype=float)
    unique_weights = np.asarray(unique_weights, dtype=float)

    psi_mono = pava_increasing(unique_psi, unique_weights)
    psi_mono = np.maximum(psi_mono, 1e-12)

    # Enforce psi_0(1)=1 after isotonic smoothing by inserting/replacing anchor.
    if not np.any(np.isclose(unique_gamma, 1.0)):
        unique_gamma = np.append(unique_gamma, 1.0)
        psi_mono = np.append(psi_mono, 1.0)

    order = np.argsort(unique_gamma)
    unique_gamma = unique_gamma[order]
    psi_mono = psi_mono[order]

    idx_anchor = np.argmin(np.abs(unique_gamma - 1.0))
    psi_mono[idx_anchor] = 1.0

    # Re-apply monotonicity around the anchor.
    psi_mono[:idx_anchor + 1] = np.minimum(psi_mono[:idx_anchor + 1], 1.0)
    psi_mono[idx_anchor:] = np.maximum(psi_mono[idx_anchor:], 1.0)
    psi_mono = pava_increasing(psi_mono)

    def psi0(gamma):
        gamma = np.asarray(gamma, dtype=float)
        return np.interp(
            gamma,
            unique_gamma,
            psi_mono,
            left=psi_mono[0],
            right=psi_mono[-1],
        )

    return psi0, unique_gamma, psi_mono, gamma_pre_all, gamma_post_all


# ==========================================================
# MODEL DYNAMICS
# ==========================================================

def make_phi_function(gamma_grid, phi_grid):
    gamma_grid = np.asarray(gamma_grid, dtype=float)
    phi_grid = np.asarray(phi_grid, dtype=float)

    def phi_func(gamma):
        gamma = np.asarray(gamma, dtype=float)
        return np.interp(
            gamma,
            gamma_grid,
            phi_grid,
            left=phi_grid[0],
            right=phi_grid[-1],
        )

    return phi_func


def one_step_lorenz(y_now, beta, phi_func, psi0):
    """
    Explicit Euler step for the Lorenz dynamics:
        dy/dt = r/(1-beta) * [int_0^x q(w)dw - y(x)*int_0^1 q(w)dw],
        q = (1 - phi(gamma))*psi0(gamma), gamma = y_x.
    """
    shares_now = np.diff(y_now)
    gamma_now = shares_now / DX

    phi_vals = phi_func(gamma_now)
    phi_vals = np.clip(phi_vals, beta + EPS, PHI_UPPER)

    psi_vals = psi0(gamma_now)
    psi_vals = np.maximum(psi_vals, 1e-12)

    q_vals = (1.0 - phi_vals) * psi_vals

    A_vals = DX * np.cumsum(q_vals)
    B_val = DX * np.sum(q_vals)

    lambda_coef = R_RATE * DT / (1.0 - beta)
    rhs = lambda_coef * (A_vals - y_now[1:] * B_val)

    raw_next = y_now[1:] + rhs
    y_next, repair_norm = project_to_lorenz_from_interior(raw_next)

    return y_next, repair_norm


def simulate_trajectory(model_data, beta, gamma_grid, phi_grid, psi0,
                        years_subset=None):
    phi_func = make_phi_function(gamma_grid, phi_grid)

    if years_subset is None:
        model_years = sorted(model_data.pre_curves.keys())
    else:
        model_years = sorted([y for y in years_subset if y in model_data.pre_curves])

    y_model = {model_years[0]: model_data.pre_curves[model_years[0]]["y"].copy()}
    repair_by_step = {}

    for y0, y1 in zip(model_years[:-1], model_years[1:]):
        if y1 != y0 + 1:
            # Restart at observed curve if years are not adjacent.
            y_model[y1] = model_data.pre_curves[y1]["y"].copy()
            continue

        y_next, repair_norm = one_step_lorenz(y_model[y0], beta, phi_func, psi0)
        y_model[y1] = y_next
        repair_by_step[(y0, y1)] = repair_norm

    return y_model, repair_by_step


def trajectory_loss(model_data, beta, gamma_grid, phi_grid, psi0,
                    years_subset=None, return_details=False):
    y_model, repair_by_step = simulate_trajectory(
        model_data, beta, gamma_grid, phi_grid, psi0, years_subset=years_subset
    )

    J = 0.0
    sum_y_obs = 0.0
    rows = []

    for year, y_mod in y_model.items():
        y_obs = model_data.pre_curves[year]["y"]
        err = y_mod[1:] - y_obs[1:]
        year_sse = float(np.sum(err ** 2))

        J += year_sse
        sum_y_obs += float(np.sum(y_obs[1:]))

        rows.append({
            "year": year,
            "sse": year_sse,
            "max_abs_error": float(np.max(np.abs(err))),
        })

    repair_values = np.array(list(repair_by_step.values()), dtype=float)
    repair_penalty = float(np.sum(repair_values ** 2)) if len(repair_values) else 0.0

    # Smoothness penalty for phi. Since gamma_grid is uniform, second differences are enough.
    smooth_penalty = float(np.sum(np.diff(phi_grid, n=2) ** 2))

    # Mild endpoint penalty: rich agents should be at least as patient as poor agents.
    endpoint_penalty = float((phi_grid[0] - phi_grid[-1]) ** 2)

    total = (
        J
        + LAMBDA_REPAIR * repair_penalty
        + LAMBDA_SMOOTH_PHI * smooth_penalty
        + LAMBDA_ENDPOINT * endpoint_penalty
    )

    Q = np.sqrt(J) / max(sum_y_obs, EPS)

    diagnostics = pd.DataFrame(rows)
    diagnostics["beta"] = beta
    diagnostics["repair_mean"] = float(np.mean(repair_values)) if len(repair_values) else 0.0
    diagnostics["repair_max"] = float(np.max(repair_values)) if len(repair_values) else 0.0

    if return_details:
        return total, J, Q, y_model, diagnostics, repair_values

    return total


# ==========================================================
# ESTIMATION OF phi(gamma)
# ==========================================================

def initial_phi_grid(beta, n):
    """
    Initial decreasing curve in admissible domain:
    poorer groups: higher phi; richer groups: lower phi.
    """
    low = beta + 0.03 * (PHI_UPPER - beta)
    high = beta + 0.85 * (PHI_UPPER - beta)

    if high <= low:
        high = beta + 0.8 * (PHI_UPPER - beta)
        low = beta + 0.05 * (PHI_UPPER - beta)

    return np.linspace(high, low, n)


def build_gamma_grid(model_data):
    all_gamma = []
    for year in model_data.years:
        all_gamma.extend(model_data.pre_curves[year]["gamma"])

    all_gamma = np.asarray(all_gamma, dtype=float)
    g_min = max(float(np.min(all_gamma)), 1e-6)
    g_max = float(np.max(all_gamma))

    # Include gamma=1 by construction if it lies inside the range.
    gamma_grid = np.linspace(g_min, g_max, N_GAMMA_GRID)

    if g_min < 1.0 < g_max and not np.any(np.isclose(gamma_grid, 1.0)):
        gamma_grid = np.sort(np.unique(np.concatenate([gamma_grid, [1.0]])))

    return gamma_grid


def estimate_phi_for_beta(model_data, beta, gamma_grid, psi0, years_subset=None):
    lb = beta + EPS
    ub = PHI_UPPER

    if lb >= ub:
        raise ValueError(f"Invalid beta={beta}; beta must be lower than {PHI_UPPER}.")

    x0 = initial_phi_grid(beta, len(gamma_grid))
    x0 = np.clip(x0, lb, ub)
    x0 = pava_decreasing(x0)
    x0 = np.clip(x0, lb, ub)

    bounds = [(lb, ub)] * len(gamma_grid)

    constraints = []

    # phi must be nonincreasing in gamma:
    # phi[i] - phi[i+1] >= 0
    for i in range(len(gamma_grid) - 1):
        constraints.append({
            "type": "ineq",
            "fun": lambda x, i=i: x[i] - x[i + 1],
        })

    def objective(x):
        x = np.asarray(x, dtype=float)
        return trajectory_loss(
            model_data=model_data,
            beta=beta,
            gamma_grid=gamma_grid,
            phi_grid=x,
            psi0=psi0,
            years_subset=years_subset,
            return_details=False,
        )

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": MAXITER, "ftol": FTOL, "disp": False},
    )

    phi_hat = np.asarray(res.x, dtype=float)
    phi_hat = np.clip(phi_hat, lb, ub)
    phi_hat = pava_decreasing(phi_hat)
    phi_hat = np.clip(phi_hat, lb, ub)

    _, J, Q, y_model, diagnostics, repair_values = trajectory_loss(
        model_data=model_data,
        beta=beta,
        gamma_grid=gamma_grid,
        phi_grid=phi_hat,
        psi0=psi0,
        years_subset=years_subset,
        return_details=True,
    )

    return EstimationResult(
        beta=float(beta),
        gamma_grid=gamma_grid.copy(),
        phi_grid=phi_hat,
        J=float(J),
        Q=float(Q),
        repair_mean=float(np.mean(repair_values)) if len(repair_values) else 0.0,
        repair_max=float(np.max(repair_values)) if len(repair_values) else 0.0,
        success=bool(res.success),
        message=str(res.message),
        y_model=y_model,
        diagnostics=diagnostics,
    )


def select_beta(model_data, gamma_grid, psi0):
    all_results = []

    for beta in BETA_LIST:
        print(f"\nRunning global estimation for beta = {beta:.2f}")

        try:
            result = estimate_phi_for_beta(model_data, beta, gamma_grid, psi0)
        except Exception as exc:
            warnings.warn(f"beta={beta:.2f} failed: {exc}")
            continue

        print(
            f"beta={beta:.2f}, Q={result.Q:.6e}, "
            f"J={result.J:.6e}, repair_mean={result.repair_mean:.3e}, "
            f"success={result.success}"
        )

        all_results.append(result)

    if not all_results:
        raise RuntimeError("No beta estimation completed successfully.")

    beta_results_df = pd.DataFrame([
        {
            "beta": r.beta,
            "J": r.J,
            "Q": r.Q,
            "repair_mean": r.repair_mean,
            "repair_max": r.repair_max,
            "success": r.success,
            "message": r.message,
        }
        for r in all_results
    ])

    best_result = min(all_results, key=lambda r: r.Q)

    return best_result, all_results, beta_results_df


def estimate_period_phis(model_data, gamma_grid, psi0, best_beta):
    period_results = {}

    for start, end in PERIODS:
        years_subset = [y for y in model_data.years if start <= y <= end]

        adjacent_count = sum(
            1 for y in years_subset
            if y + 1 in years_subset and y + 1 in model_data.pre_curves
        )

        if len(years_subset) < 2 or adjacent_count == 0:
            print(f"Skipping period {start}-{end}: not enough adjacent data.")
            continue

        print(f"\nEstimating period-specific phi for {start}-{end}, beta={best_beta:.2f}")

        result = estimate_phi_for_beta(
            model_data=model_data,
            beta=best_beta,
            gamma_grid=gamma_grid,
            psi0=psi0,
            years_subset=years_subset,
        )

        period_results[(start, end)] = result

        print(
            f"period {start}-{end}, Q={result.Q:.6e}, "
            f"repair_mean={result.repair_mean:.3e}, success={result.success}"
        )

    return period_results


# ==========================================================
# PLOTS
# ==========================================================

def plot_pre_lorenz_curves(model_data):
    plt.figure(figsize=(8, 6))

    cmap = LinearSegmentedColormap.from_list("blue_red", ["#4aa3ff", "#ff4a4a"])
    norm = plt.Normalize(min(model_data.years), max(model_data.years))

    for year in model_data.years:
        shares_pre = model_data.pre_curves[year]["shares"]
        x_smooth, y_smooth = smooth_lorenz_curve(shares_pre)

        plt.plot(
            x_smooth,
            y_smooth,
            color=cmap(norm(year)),
            linewidth=1.8,
            alpha=0.9,
        )

    plt.plot([0, 1], [0, 1], "--", linewidth=1.5, color="black")
    plt.plot([], [], color=cmap(norm(min(model_data.years))), label=str(min(model_data.years)))
    plt.plot([], [], color=cmap(norm(max(model_data.years))), label=str(max(model_data.years)))

    plt.title(f"Lorenz curves, pre-tax ({COUNTRY_NAME}, {START_YEAR}â{END_YEAR})")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y(x,t)$")
    plt.grid(alpha=0.3)
    plt.legend(title="Year")
    save_or_show("lorenz_pre_tax.png")


def plot_psi0(unique_gamma, psi_mono, gamma_pre_all, gamma_post_all):
    gamma_dense = np.linspace(unique_gamma.min(), unique_gamma.max(), 1000)
    psi_dense = np.interp(gamma_dense, unique_gamma, psi_mono)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        gamma_pre_all,
        gamma_post_all,
        s=18,
        alpha=0.15,
        label="Data points",
    )

    plt.plot(
        gamma_dense,
        psi_dense,
        linewidth=3,
        label=r"$\psi_0(\gamma)$",
    )

    plt.plot(
        gamma_dense,
        gamma_dense,
        "--",
        linewidth=1.2,
        label=r"$\psi_0(\gamma)=\gamma$",
    )

    plt.scatter([1.0], [1.0], s=60, color="black", label=r"anchor $(1,1)$")

    plt.title(rf"Redistribution function $\psi_0(\gamma)$, {COUNTRY_NAME}")
    plt.xlabel(r"$\gamma^{pre}$")
    plt.ylabel(r"$\gamma^{post}$")
    plt.grid(alpha=0.3)
    plt.legend()
    save_or_show("psi0.png")


def plot_beta_selection(beta_results_df):
    plt.figure(figsize=(8, 5))

    plt.plot(
        beta_results_df["beta"],
        beta_results_df["Q"],
        marker="o",
        linewidth=2,
    )

    plt.title(rf"Selection of $\beta$ by trajectory criterion, {COUNTRY_NAME}")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$Q=\sqrt{J}/\sum y_j^{{obs}}$")
    plt.grid(alpha=0.3)
    save_or_show("beta_selection.png")


def plot_phi_star(best_result):
    plt.figure(figsize=(10, 6))

    plt.plot(
        best_result.gamma_grid,
        best_result.phi_grid,
        color="black",
        linewidth=4,
        label=rf"$\varphi^*(\gamma)$, $\beta={best_result.beta:.2f}$",
    )

    plt.axhline(
        best_result.beta,
        linestyle="--",
        color="black",
        linewidth=2,
        alpha=0.8,
        label=r"$\beta$",
    )

    plt.axhline(
        1.0,
        linestyle=":",
        color="black",
        linewidth=2,
        alpha=0.8,
        label=r"$1$",
    )

    plt.title(rf"Recovered global $\varphi^*(\gamma)$, {COUNTRY_NAME}")
    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$\varphi(\gamma)$")
    plt.grid(alpha=0.3)
    plt.legend()
    save_or_show("phi_star.png")


def plot_rho_star(best_result):
    plt.figure(figsize=(10, 6))

    rho_star = R_RATE * best_result.phi_grid

    plt.plot(
        best_result.gamma_grid,
        rho_star,
        color="black",
        linewidth=4,
        label=rf"$\rho^*(\gamma)=r\varphi^*(\gamma)$",
    )

    plt.title(
        rf"Discount rate $\rho(\gamma)=r\varphi(\gamma)$, "
        rf"{COUNTRY_NAME}; $r={R_RATE:.3f}$"
    )
    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$\rho(\gamma)$")
    plt.grid(alpha=0.3)
    plt.legend()
    save_or_show("rho_star.png")


def plot_period_phis(period_results, best_beta):
    if not period_results:
        return

    plt.figure(figsize=(10, 6))

    for (start, end), result in period_results.items():
        plt.plot(
            result.gamma_grid,
            result.phi_grid,
            linewidth=3,
            label=f"{start}â{end}",
        )

    plt.axhline(
        best_beta,
        linestyle="--",
        color="black",
        linewidth=2,
        alpha=0.8,
        label=r"$\beta$",
    )

    plt.title(rf"Period-specific $\varphi(\gamma)$, {COUNTRY_NAME}")
    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$\varphi(\gamma)$")
    plt.grid(alpha=0.3)
    plt.legend()
    save_or_show("phi_periods.png")


def plot_period_rhos(period_results):
    if not period_results:
        return

    plt.figure(figsize=(10, 6))

    for (start, end), result in period_results.items():
        rho_vals = R_RATE * result.phi_grid
        plt.plot(
            result.gamma_grid,
            rho_vals,
            linewidth=3,
            label=f"{start}â{end}",
        )

    plt.title(rf"Period-specific $\rho(\gamma)=r\varphi(\gamma)$, {COUNTRY_NAME}")
    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$\rho(\gamma)$")
    plt.grid(alpha=0.3)
    plt.legend()
    save_or_show("rho_periods.png")


def plot_observed_vs_model(model_data, best_result):
    years_to_plot = [1980, 1990, 2000, 2010, 2020, 2024]

    plt.figure(figsize=(9, 6))

    for year in years_to_plot:
        if year in best_result.y_model and year in model_data.pre_curves:
            x = model_data.pre_curves[year]["x"]
            y_obs = model_data.pre_curves[year]["y"]
            y_mod = best_result.y_model[year]

            plt.plot(x, y_obs, linewidth=2, label=f"observed {year}")
            plt.plot(x, y_mod, "--", linewidth=2, label=f"model {year}")

    plt.plot([0, 1], [0, 1], "--", color="black")

    plt.title(rf"Observed vs model Lorenz curves, {COUNTRY_NAME}, $\beta={best_result.beta:.2f}$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    save_or_show("observed_vs_model.png")


# ==========================================================
# SAVE RESULTS
# ==========================================================

def save_results(beta_results_df, best_result, period_results,
                 unique_gamma, psi_mono):
    beta_results_df.to_csv(
        os.path.join(OUTPUT_DIR, f"beta_selection_{COUNTRY_NAME.lower()}_corrected.csv"),
        index=False,
    )

    phi_save_df = pd.DataFrame({
        "country": COUNTRY_NAME,
        "beta": best_result.beta,
        "r_rate": R_RATE,
        "r_rate_label": R_RATE_LABEL,
        "r_rate_date": R_RATE_DATE,
        "gamma": best_result.gamma_grid,
        "phi_star": best_result.phi_grid,
        "rho_star": R_RATE * best_result.phi_grid,
    })

    phi_save_df.to_csv(
        os.path.join(OUTPUT_DIR, f"phi_rho_star_{COUNTRY_NAME.lower()}_corrected.csv"),
        index=False,
    )

    best_result.diagnostics.to_csv(
        os.path.join(OUTPUT_DIR, f"trajectory_diagnostics_{COUNTRY_NAME.lower()}_corrected.csv"),
        index=False,
    )

    psi_save_df = pd.DataFrame({
        "country": COUNTRY_NAME,
        "gamma_pre": unique_gamma,
        "psi_0": psi_mono,
    })

    psi_save_df.to_csv(
        os.path.join(OUTPUT_DIR, f"psi0_{COUNTRY_NAME.lower()}_corrected.csv"),
        index=False,
    )

    period_rows = []

    for (start, end), result in period_results.items():
        for gamma_val, phi_val in zip(result.gamma_grid, result.phi_grid):
            period_rows.append({
                "country": COUNTRY_NAME,
                "beta": best_result.beta,
                "r_rate": R_RATE,
                "r_rate_label": R_RATE_LABEL,
                "r_rate_date": R_RATE_DATE,
                "period_start": start,
                "period_end": end,
                "period": f"{start}-{end}",
                "gamma": gamma_val,
                "phi_period": phi_val,
                "rho_period": R_RATE * phi_val,
                "Q_period": result.Q,
                "repair_mean": result.repair_mean,
                "repair_max": result.repair_max,
                "success": result.success,
                "message": result.message,
            })

    period_df = pd.DataFrame(period_rows)
    period_df.to_csv(
        os.path.join(OUTPUT_DIR, f"phi_rho_periods_{COUNTRY_NAME.lower()}_corrected.csv"),
        index=False,
    )

    summary = {
        "country": COUNTRY_NAME,
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "r_rate": R_RATE,
        "r_rate_label": R_RATE_LABEL,
        "r_rate_date": R_RATE_DATE,
        "best_beta": best_result.beta,
        "best_J": best_result.J,
        "best_Q": best_result.Q,
        "repair_mean": best_result.repair_mean,
        "repair_max": best_result.repair_max,
    }

    pd.DataFrame([summary]).to_csv(
        os.path.join(OUTPUT_DIR, f"summary_{COUNTRY_NAME.lower()}_corrected.csv"),
        index=False,
    )

    print("\nSaved files in:", os.path.abspath(OUTPUT_DIR))
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        print(" -", fname)


# ==========================================================
# MAIN
# ==========================================================

def main():
    ensure_output_dir(OUTPUT_DIR)

    print("=" * 60)
    print("Corrected inverse problem estimation")
    print("=" * 60)
    print(f"Country: {COUNTRY_NAME}")
    print(f"R_RATE = {R_RATE} ({R_RATE_LABEL}, fixed at {R_RATE_DATE})")
    print(f"Data file: {FILE_PATH}")

    df = load_dataframe(FILE_PATH)
    model_data = build_model_data(df)

    plot_pre_lorenz_curves(model_data)

    psi0, unique_gamma, psi_mono, gamma_pre_all, gamma_post_all = build_psi0(model_data)
    plot_psi0(unique_gamma, psi_mono, gamma_pre_all, gamma_post_all)

    gamma_grid = build_gamma_grid(model_data)

    best_result, all_results, beta_results_df = select_beta(
        model_data=model_data,
        gamma_grid=gamma_grid,
        psi0=psi0,
    )

    print("\n" + "=" * 60)
    print("BEST RESULT")
    print("=" * 60)
    print(f"Best beta: {best_result.beta:.2f}")
    print(f"Best J:    {best_result.J:.8e}")
    print(f"Best Q:    {best_result.Q:.8e}")
    print(f"Repair mean: {best_result.repair_mean:.8e}")
    print(f"Repair max:  {best_result.repair_max:.8e}")
    print(f"Success: {best_result.success}")
    print(f"Message: {best_result.message}")

    period_results = estimate_period_phis(
        model_data=model_data,
        gamma_grid=gamma_grid,
        psi0=psi0,
        best_beta=best_result.beta,
    )

    plot_beta_selection(beta_results_df)
    plot_phi_star(best_result)
    plot_rho_star(best_result)
    plot_period_phis(period_results, best_result.beta)
    plot_period_rhos(period_results)
    plot_observed_vs_model(model_data, best_result)

    save_results(
        beta_results_df=beta_results_df,
        best_result=best_result,
        period_results=period_results,
        unique_gamma=unique_gamma,
        psi_mono=psi_mono,
    )


if __name__ == "__main__":
    main()