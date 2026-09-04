import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from matplotlib.colors import LinearSegmentedColormap


# ==========================================================
# НАСТРОЙКИ
# ==========================================================

FILE_PATH = "/Users/ulianaalekseeva/PyCharmMiscProject/sweden_income_merged_filled.csv"

START_YEAR = 1980
END_YEAR = 2020

DECILES = [
    "p0p10", "p10p20", "p20p30", "p30p40", "p40p50",
    "p50p60", "p60p70", "p70p80", "p80p90", "p90p100"
]

DX = 0.1
DT = 1.0

R_RATE = 0.034
BETA_MODEL = 0.7

EPS_PHI = 1e-8

LAMBDA_REG = 1e-6
LAMBDA_SMOOTH = 1e-5

SMOOTH_WINDOW = 31
SMOOTH_POLYORDER = 2

plt.rcParams["ps.fonttype"] = 42
plt.rcParams["pdf.fonttype"] = 42


# ==========================================================
# ЗАГРУЗКА ДАННЫХ
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
# ФУНКЦИИ
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
# 1. КРИВЫЕ ЛОРЕНЦА
# ==========================================================

years_list = []

for year in range(START_YEAR, END_YEAR + 1):
    values_pre = get_year_values(df, "pre", year)
    if values_pre is not None:
        years_list.append(year)

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

plt.title(f"Кривые Лоренца ({START_YEAR}–{END_YEAR})")
plt.xlabel(r"$x$")
plt.ylabel(r"$y(x,t)$")
plt.grid(alpha=0.3)
plt.legend(title="Год")
plt.tight_layout()
plt.show()


# ==========================================================
# 2. ПОСТРОЕНИЕ psi_0
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
plt.scatter(gamma_pre_all, gamma_post_all, s=18, alpha=0.18, label="Точки данных")
plt.plot(gamma_dense, psi_dense, linewidth=3, label=r"$\psi_0(\gamma)$")
plt.plot(gamma_dense, gamma_dense, "--", linewidth=1.2, label=r"$\psi_0(\gamma)=\gamma$")
plt.title(r"Функция $\psi_0(\gamma)$")
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$\psi_0(\gamma)$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# 3. PRE-TAX КРИВЫЕ ПО ГОДАМ
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

print("Число переходов:", len(all_transition_years))
print(all_transition_years[:10], "...")

lambda_coef = R_RATE * DT / (1.0 - BETA_MODEL)


# ==========================================================
# 4. ВОССТАНОВЛЕНИЕ phi ДЛЯ КАЖДОГО ГОДА
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

    # убывание именно по gamma
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

        return sse

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

    # чистый one-step без регуляризации
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
print("РЕЗУЛЬТАТЫ ПО ГОДАМ")
print("==============================")
print(year_results_df.head(15))
print("\nСредний one-step J по годам:", year_results_df["J_one"].mean())



# ==========================================================
# 6. СГЛАЖИВАНИЕ ГОДОВЫХ phi НА ОБЩЕЙ СЕТКЕ
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
# 7. MEAN И MEDIAN ПО ГОДОВЫМ ФУНКЦИЯМ
# ==========================================================

phi_mean = np.mean(all_phi_smooth, axis=0)
phi_median = np.median(all_phi_smooth, axis=0)

plt.figure(figsize=(10, 6))

for phi_smooth in all_phi_smooth:
    plt.plot(
        common_gamma,
        phi_smooth,
        linewidth=1.5,
        alpha=0.45
    )

plt.plot(
    common_gamma,
    phi_mean,
    color="black",
    linewidth=4,
    label="mean"
)


plt.axhline(
    BETA_MODEL,
    linestyle="--",
    color="black",
    linewidth=2,
    alpha=0.8,
    label=r"$\beta$"
)

plt.title(rf"Итоговая функция $\phi^*(\gamma)$, $\beta={BETA_MODEL}$")
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$\phi$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# 8. АППРОКСИМАЦИЯ MEAN ЯВНОЙ ФОРМУЛОЙ
# ==========================================================

def phi_exp(gamma, params):
    A, c = params
    return BETA_MODEL + A * np.exp(-c * gamma)


def phi_power(gamma, params):
    A, c, p = params
    return BETA_MODEL + A / ((1.0 + c * gamma) ** p)


def phi_stretched_exp(gamma, params):
    A, c, p = params
    return BETA_MODEL + A * np.exp(-c * gamma ** p)


def fit_model(model_func, x0, bounds):
    def objective(params):
        phi_fit = model_func(common_gamma, params)
        return np.mean((phi_fit - phi_mean) ** 2)

    return minimize(
        objective,
        x0=np.array(x0, dtype=float),
        bounds=bounds,
        method="L-BFGS-B"
    )


A0 = max(phi_mean[0] - BETA_MODEL, 1e-3)

res_exp = fit_model(
    phi_exp,
    x0=[A0, 0.5],
    bounds=[(1e-8, 10.0), (1e-8, 20.0)]
)

res_power = fit_model(
    phi_power,
    x0=[A0, 0.5, 1.0],
    bounds=[(1e-8, 10.0), (1e-8, 20.0), (1e-8, 20.0)]
)

res_stretched = fit_model(
    phi_stretched_exp,
    x0=[A0, 0.5, 1.0],
    bounds=[(1e-8, 10.0), (1e-8, 20.0), (0.1, 5.0)]
)

phi_fit_exp = phi_exp(common_gamma, res_exp.x)
phi_fit_power = phi_power(common_gamma, res_power.x)
phi_fit_stretched = phi_stretched_exp(common_gamma, res_stretched.x)

err_exp = np.mean((phi_fit_exp - phi_mean) ** 2)
err_power = np.mean((phi_fit_power - phi_mean) ** 2)
err_stretched = np.mean((phi_fit_stretched - phi_mean) ** 2)

fits = {
    "exp": {
        "err": err_exp,
        "values": phi_fit_exp,
        "params": res_exp.x,
        "func": phi_exp
    },
    "power": {
        "err": err_power,
        "values": phi_fit_power,
        "params": res_power.x,
        "func": phi_power
    },
    "stretched exp": {
        "err": err_stretched,
        "values": phi_fit_stretched,
        "params": res_stretched.x,
        "func": phi_stretched_exp
    }
}

best_name = min(fits, key=lambda name: fits[name]["err"])
best_params = fits[best_name]["params"]
best_func = fits[best_name]["func"]
best_fit = fits[best_name]["values"]

print("\n==============================")
print("АППРОКСИМАЦИЯ MEAN phi")
print("==============================")
print(f"exp MSE           = {err_exp:.8f}")
print(f"power MSE         = {err_power:.8f}")
print(f"stretched exp MSE = {err_stretched:.8f}")
print()
print(f"Лучшая аппроксимация: {best_name}")

if best_name == "exp":
    A, c = best_params
    print(f"phi(gamma) = {BETA_MODEL:.4f} + {A:.6f} * exp(-{c:.6f} * gamma)")

elif best_name == "power":
    A, c, p = best_params
    print(f"phi(gamma) = {BETA_MODEL:.4f} + {A:.6f} / (1 + {c:.6f} * gamma)^{p:.6f}")

else:
    A, c, p = best_params
    print(f"phi(gamma) = {BETA_MODEL:.4f} + {A:.6f} * exp(-{c:.6f} * gamma^{p:.6f})")


plt.figure(figsize=(10, 6))

plt.plot(
    common_gamma,
    phi_mean,
    color="black",
    linewidth=4,
    label="mean"
)

plt.plot(
    common_gamma,
    phi_fit_exp,
    linewidth=2.5,
    linestyle="--",
    label="exp"
)

plt.plot(
    common_gamma,
    phi_fit_power,
    linewidth=2.5,
    linestyle="--",
    label="power"
)

plt.plot(
    common_gamma,
    phi_fit_stretched,
    linewidth=2.5,
    linestyle="--",
    label="stretched exp"
)

plt.axhline(
    BETA_MODEL,
    linestyle="--",
    color="black",
    linewidth=2,
    alpha=0.8,
    label=r"$\beta$"
)

plt.title(rf"Аппроксимация средней функции $\phi^*(\gamma)$")
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$\phi$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))

plt.plot(
    common_gamma,
    phi_mean,
    color="black",
    linewidth=4,
    label="mean"
)

plt.plot(
    common_gamma,
    best_fit,
    color="red",
    linewidth=3,
    linestyle="--",
    label=f"best: {best_name}"
)

plt.axhline(
    BETA_MODEL,
    linestyle="--",
    color="black",
    linewidth=2,
    alpha=0.8,
    label=r"$\beta$"
)

plt.title(rf"Явная аппроксимация $\phi^*(\gamma)$")
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$\phi$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# 9. ДИНАМИКА С АППРОКСИМИРОВАННОЙ phi
# ==========================================================

def phi_approx(gamma):
    gamma = np.asarray(gamma, dtype=float)
    return best_func(gamma, best_params)


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

    y_next_interior = y_now[1:] + rhs

    return project_to_lorenz(y_next_interior)


model_years = sorted(pre_curves.keys())

y_model_approx = {
    model_years[0]: pre_curves[model_years[0]]["y"].copy()
}

for year in model_years[:-1]:
    next_year = year + 1

    if next_year in pre_curves:
        y_model_approx[next_year] = one_step_with_phi(
            y_model_approx[year],
            phi_approx
        )


# ==========================================================
# 10. ФУНКЦИОНАЛЫ ДЛЯ АППРОКСИМИРОВАННОЙ phi
# ==========================================================

J_one_approx = 0.0
n_one_approx = 0

for year0, year1 in all_transition_years:
    y0 = pre_curves[year0]["y"]
    y1 = pre_curves[year1]["y"]

    y_model_next = one_step_with_phi(y0, phi_approx)

    J_one_approx += np.sum((y_model_next[1:] - y1[1:]) ** 2)
    n_one_approx += len(y1[1:])

MSE_one_approx = J_one_approx / n_one_approx
RMSE_one_approx = np.sqrt(MSE_one_approx)


J_traj_approx = 0.0
n_traj_approx = 0

years_error_approx = []
errors_year_approx = []

for year in sorted(y_model_approx.keys()):
    y_model = y_model_approx[year]
    y_obs = pre_curves[year]["y"]

    err_year = np.sum((y_model[1:] - y_obs[1:]) ** 2) / 10

    years_error_approx.append(year)
    errors_year_approx.append(err_year)

    J_traj_approx += np.sum((y_model[1:] - y_obs[1:]) ** 2)
    n_traj_approx += len(y_obs[1:])

MSE_traj_approx = J_traj_approx / n_traj_approx
RMSE_traj_approx = np.sqrt(MSE_traj_approx)

print("\n==============================")
print("ФУНКЦИОНАЛЫ ДЛЯ АППРОКСИМИРОВАННОЙ phi")
print("==============================")
print(f"Тип аппроксимации: {best_name}")
print()
print(f"One-step J    = {J_one_approx:.6f}")
print(f"One-step MSE  = {MSE_one_approx:.6f}")
print(f"One-step RMSE = {RMSE_one_approx:.6f}")
print()
print(f"Trajectory J    = {J_traj_approx:.6f}")
print(f"Trajectory MSE  = {MSE_traj_approx:.6f}")
print(f"Trajectory RMSE = {RMSE_traj_approx:.6f}")


# ==========================================================
# 11. ГРАФИК ОШИБКИ ПО ГОДАМ
# ==========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    years_error_approx,
    errors_year_approx,
    marker="o",
    linewidth=2
)

plt.title("Ошибка модели по годам для аппроксимированной функции phi")
plt.xlabel("Год")
plt.ylabel("Средняя сумма квадратов ошибок")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================================================
# 12. СРАВНЕНИЕ МОДЕЛЬНЫХ И НАБЛЮДАЕМЫХ КРИВЫХ ЛОРЕНЦА
# ==========================================================

years_to_plot = [1980, 1990, 2000, 2010, 2020]

plt.figure(figsize=(9, 6))

for year in years_to_plot:
    if year in pre_curves and year in y_model_approx:
        x = pre_curves[year]["x"]
        y_obs = pre_curves[year]["y"]
        y_mod = y_model_approx[year]

        plt.plot(
            x,
            y_obs,
            linewidth=2,
            alpha=0.7,
            label=f"obs {year}"
        )

        plt.plot(
            x,
            y_mod,
            "--",
            linewidth=2,
            alpha=0.9,
            label=f"model {year}"
        )

plt.plot([0, 1], [0, 1], "--", color="black", linewidth=1.2)

plt.title("Наблюдаемые и модельные кривые Лоренца")
plt.xlabel(r"$x$")
plt.ylabel(r"$y(x,t)$")
plt.grid(alpha=0.3)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

# ==========================================================
# 17. СОХРАНЕНИЕ НЕАППРОКСИМИРОВАННОЙ phi* ДЛЯ СРАВНЕНИЯ СТРАН
# ==========================================================

COUNTRY_NAME = "Sweden"

phi_save_df = pd.DataFrame({
    "country": COUNTRY_NAME,
    "gamma": common_gamma,
    "phi_star_raw": phi_mean,
    "beta": BETA_MODEL,
})

phi_save_df.to_csv(
    f"test_phi_star_raw_{COUNTRY_NAME.lower()}.csv",
    index=False
)

print(f"\nСохранено: phi_star_raw_{COUNTRY_NAME.lower()}.csv")