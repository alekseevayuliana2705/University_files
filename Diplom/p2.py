import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==========================================================
# НАСТРОЙКИ
# ==========================================================

FILE_PATH = "/Users/ulianaalekseeva/PyCharmMiscProject/germany_income_merged_filled.csv"
START_YEAR = 1980
END_YEAR = 2020

DECILES = [
    "p0p10", "p10p20", "p20p30", "p30p40", "p40p50",
    "p50p60", "p60p70", "p70p80", "p80p90", "p90p100"
]

DX = 0.1
R_RATE = 0.03
BETA_MODEL = 0.7
LAMBDA_REG = 1e-3
WINDOW_SIZE = 5

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
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
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

def pava_increasing(y, w=None):
    y = np.asarray(y, dtype=float)
    n = len(y)

    if w is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(w, dtype=float)

    blocks = []
    for i in range(n):
        blocks.append([i, i, y[i], w[i]])

        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            b2 = blocks.pop()
            b1 = blocks.pop()
            total_w = b1[3] + b2[3]
            avg = (b1[2] * b1[3] + b2[2] * b2[3]) / total_w
            blocks.append([b1[0], b2[1], avg, total_w])

    out = np.empty(n, dtype=float)
    for start, end, avg, _ in blocks:
        out[start:end + 1] = avg

    return out

# ==========================================================
# 1. PRE-TAX КРИВЫЕ ЛОРЕНЦА
# ==========================================================

plt.figure(figsize=(8, 6))
valid_lorenz_years = []

for year in range(START_YEAR, END_YEAR + 1):
    values_pre = get_year_values(df, "pre", year)
    if values_pre is None:
        continue

    shares_pre = shares_from_values(values_pre)
    x, y = lorenz_from_shares(shares_pre)
    plt.plot(x, y, linewidth=1, alpha=0.65)
    valid_lorenz_years.append(year)

plt.plot([0, 1], [0, 1], "--", linewidth=1.5, label="Линия равенства")
plt.title(f"Кривые Лоренца до налогов ({START_YEAR}–{END_YEAR})")
plt.xlabel("Накопленная доля населения")
plt.ylabel("Накопленная доля дохода")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("Число лет для pre-tax кривых Лоренца:", len(valid_lorenz_years))

# ==========================================================
# 2. СТРОИМ psi_0
# ==========================================================

gamma_pre_all = []
gamma_post_all = []
used_years = []

for year in range(START_YEAR, END_YEAR + 1):
    values_pre = get_year_values(df, "pre", year)
    values_post = get_year_values(df, "post", year)

    if values_pre is None or values_post is None:
        continue

    shares_pre = shares_from_values(values_pre)
    shares_post = shares_from_values(values_post)

    gamma_pre_all.extend(gamma_from_shares(shares_pre).tolist())
    gamma_post_all.extend(gamma_from_shares(shares_post).tolist())
    used_years.append(year)

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

unique_gamma = np.array(unique_gamma)
unique_psi = np.array(unique_psi)
unique_weights = np.array(unique_weights, dtype=float)

psi_pava = pava_increasing(unique_psi, unique_weights)
psi_mono = np.maximum.accumulate(psi_pava)

def psi0(x):
    x = np.asarray(x, dtype=float)
    return np.interp(x, unique_gamma, psi_mono)

gamma_dense = np.linspace(unique_gamma.min(), unique_gamma.max(), 1000)
psi_dense = psi0(gamma_dense)

print("\nПроверка psi_0:")
print("Неубывает на узлах? ->", np.all(np.diff(psi_mono) >= -1e-12))
print("Неубывает на плотной сетке? ->", np.all(np.diff(psi_dense) >= -1e-12))

print("\nПроверка нормировки по годам:")
for year in used_years:
    values_pre = get_year_values(df, "pre", year)
    values_post = get_year_values(df, "post", year)

    shares_pre = shares_from_values(values_pre)
    shares_post = shares_from_values(values_post)

    gamma_pre = gamma_from_shares(shares_pre)
    gamma_post = gamma_from_shares(shares_post)

    empirical = np.sum(gamma_post * DX)
    model = np.sum(psi0(gamma_pre) * DX)
    print(f"{year}: empirical = {empirical:.6f}, model = {model:.6f}")

plt.figure(figsize=(8, 6))
plt.scatter(gamma_pre_all, gamma_post_all, s=18, alpha=0.18, label="Точки данных")
plt.plot(unique_gamma, unique_psi, linewidth=1.3, alpha=0.7, label="Средние значения")
plt.plot(gamma_dense, psi_dense, linewidth=3, label=r"$\psi_0(\gamma)$")
plt.plot(
    [min(gamma_dense.min(), psi_dense.min()), max(gamma_dense.max(), psi_dense.max())],
    [min(gamma_dense.min(), psi_dense.min()), max(gamma_dense.max(), psi_dense.max())],
    "--",
    linewidth=1.2,
    label=r"$\psi_0(\gamma)=\gamma$"
)
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

for year in range(START_YEAR, END_YEAR + 1):
    values_pre = get_year_values(df, "pre", year)
    if values_pre is None:
        continue

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

window_list = []
for i in range(len(all_transition_years) - WINDOW_SIZE + 1):
    window_list.append(all_transition_years[i:i + WINDOW_SIZE])

gamma_min = float(np.min([np.min(pre_curves[y]["gamma"]) for y in pre_curves]))
gamma_max = float(np.max([np.max(pre_curves[y]["gamma"]) for y in pre_curves]))

# ==========================================================
# 4. ЖЕСТКАЯ ПАРАМЕТРИЗАЦИЯ phi
# ==========================================================

def phi_from_params(gamma, params, beta_model):
    a, c = params
    gamma = np.asarray(gamma, dtype=float)
    return beta_model + np.exp(a - np.exp(c) * gamma)

def initial_params():
    return np.array([0.0, -1.0])

def window_objective(params, window):
    total_sse = 0.0

    for year0, year1 in window:
        y0 = pre_curves[year0]["y"]
        y1 = pre_curves[year1]["y"]
        gamma0 = pre_curves[year0]["gamma"]

        lhs = y1[1:] - y0[1:]

        phi_vals = phi_from_params(gamma0, params, BETA_MODEL)
        psi_vals = psi0(gamma0)

        q_vals = (1.0 - phi_vals) * psi_vals
        A_vals = DX * np.cumsum(q_vals)
        B_val = DX * np.sum(q_vals)

        rhs = (R_RATE / (1.0 - BETA_MODEL)) * (A_vals - y0[1:] * B_val)
        residual = lhs - rhs

        total_sse += np.sum(residual ** 2)

    a, c = params
    reg = LAMBDA_REG * (a ** 2 + c ** 2)

    return total_sse + reg

# ==========================================================
# 5. phi ПО ОКНАМ
# ==========================================================

phi_by_window = {}

for window in window_list:
    start_year = window[0][0]
    end_year = window[-1][1]

    result = minimize(
        window_objective,
        initial_params(),
        args=(window,),
        method="L-BFGS-B"
    )

    params_opt = result.x
    gamma_dense_phi = np.linspace(gamma_min, gamma_max, 400)
    phi_dense = phi_from_params(gamma_dense_phi, params_opt, BETA_MODEL)

    phi_by_window[(start_year, end_year)] = {
        "window": window,
        "params": params_opt,
        "gamma_dense": gamma_dense_phi,
        "phi_dense": phi_dense,
        "success": result.success,
        "fun": result.fun
    }

    a_opt, c_opt = params_opt
    print(
        f"\nОкно {start_year}->{end_year}: "
        f"success={result.success}, objective={result.fun:.6e}, "
        f"a={a_opt:.6f}, c={c_opt:.6f}, "
        f"min(phi)={phi_dense.min():.6f}, "
        f"phi>beta={np.all(phi_dense > BETA_MODEL)}, "
        f"nonincreasing={np.all(np.diff(phi_dense) <= 1e-12)}"
    )

plt.figure(figsize=(9, 6))
for (start_year, end_year), item in phi_by_window.items():
    plt.plot(
        item["gamma_dense"],
        item["phi_dense"],
        linewidth=2,
        alpha=0.8,
        label=f"{start_year}->{end_year}"
    )

plt.axhline(BETA_MODEL, linestyle="--", linewidth=1.2, label=r"$\beta$")
plt.title(r"Функции $\varphi(\gamma)$ по пятилетним окнам")
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$\varphi(\gamma)$")
plt.grid(alpha=0.3)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

# ==========================================================
# 6. СРЕДНЯЯ phi*
# ==========================================================

common_gamma = np.linspace(gamma_min, gamma_max, 400)
all_phi = []

for item in phi_by_window.values():
    phi_vals = np.interp(common_gamma, item["gamma_dense"], item["phi_dense"])
    all_phi.append(phi_vals)

all_phi = np.array(all_phi)
phi_mean = np.mean(all_phi, axis=0)
phi_std = np.std(all_phi, axis=0)

# ==========================================================
# 7. АППРОКСИМАЦИЯ СРЕДНЕЙ phi*
# ==========================================================

def phi_star_parametric(gamma, params):
    a, c = params
    gamma = np.asarray(gamma, dtype=float)
    return BETA_MODEL + np.exp(a - np.exp(c) * gamma)

def fit_objective(params):
    phi_fit = phi_star_parametric(common_gamma, params)
    return np.mean((phi_fit - phi_mean) ** 2)

res_star = minimize(fit_objective, initial_params(), method="L-BFGS-B")
params_star = res_star.x
a_star, c_star = params_star

phi_star_fit = phi_star_parametric(common_gamma, params_star)

print("\n==============================")
print("ПАРАМЕТРЫ СРЕДНЕЙ phi*")
print("==============================")
print(f"success = {res_star.success}")
print(f"message = {res_star.message}")
print(f"a* = {a_star:.6f}")
print(f"c* = {c_star:.6f}")

print("\nФОРМУЛА phi*(gamma):")
print(
    f"phi*(γ) = {BETA_MODEL:.4f} + exp("
    f"{a_star:.4f} - exp({c_star:.4f}) * γ)"
)

plt.figure(figsize=(8, 5))
plt.plot(common_gamma, phi_mean, linewidth=3, label=r"$\varphi^*$")
plt.axhline(BETA_MODEL, linestyle="--", linewidth=1.2, label=r"$\beta$")
plt.title(r"Функция $\varphi^*$")
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$\varphi(\gamma)$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================================
# 8. ФУНКЦИОНАЛ ПО ОКНАМ ДЛЯ НАЙДЕННЫХ phi
# ==========================================================

def model_next_lorenz(y_now, gamma_now, params):
    phi_vals = phi_from_params(gamma_now, params, BETA_MODEL)
    psi_vals = psi0(gamma_now)

    q_vals = (1.0 - phi_vals) * psi_vals
    A_vals = DX * np.cumsum(q_vals)
    B_val = DX * np.sum(q_vals)

    rhs = (R_RATE / (1.0 - BETA_MODEL)) * (A_vals - y_now[1:] * B_val)
    return y_now[1:] + rhs

def window_fit_functional(window, params):
    total_J = 0.0

    for year0, year1 in window:
        y0 = pre_curves[year0]["y"]
        y1 = pre_curves[year1]["y"]
        gamma0 = pre_curves[year0]["gamma"]

        y_model_next = model_next_lorenz(y0, gamma0, params)
        y_obs_next = y1[1:]

        total_J += np.sum((y_model_next - y_obs_next) ** 2)

    return total_J

print("\nЗначения функционала по окнам:")
functional_values = []

for (start_year, end_year), item in phi_by_window.items():
    J_val = window_fit_functional(item["window"], item["params"])
    functional_values.append(J_val)
    print(f"{start_year}->{end_year}: J = {J_val:.8e}")

functional_values = np.array(functional_values)

print("\nСводка по функционалу:")
print(f"min J    = {functional_values.min():.8e}")
print(f"max J    = {functional_values.max():.8e}")
print(f"mean J   = {functional_values.mean():.8e}")
print(f"median J = {np.median(functional_values):.8e}")

# ==========================================================
# 9. ДИНАМИКА С phi*
# ==========================================================

def phi_star_func(gamma):
    return phi_star_parametric(gamma, params_star)

def project_to_lorenz(y_interior):
    y_interior = np.asarray(y_interior, dtype=float)

    y_full = np.concatenate([[0.0], y_interior])
    shares = np.diff(y_full)

    shares = np.maximum(shares, 1e-12)
    shares = shares / shares.sum()

    y_proj = np.concatenate([[0.0], np.cumsum(shares)])
    y_proj[-1] = 1.0

    return y_proj

def one_model_step_phi_star(y_now):
    shares_now = np.diff(y_now)
    gamma_now = shares_now / DX

    phi_vals = phi_star_func(gamma_now)
    psi_vals = psi0(gamma_now)

    q_vals = (1.0 - phi_vals) * psi_vals
    A_vals = DX * np.cumsum(q_vals)
    B_val = DX * np.sum(q_vals)

    rhs = (R_RATE / (1.0 - BETA_MODEL)) * (A_vals - y_now[1:] * B_val)
    y_next_interior = y_now[1:] + rhs

    return project_to_lorenz(y_next_interior)

model_years = sorted(pre_curves.keys())
if len(model_years) < 2:
    raise ValueError("Недостаточно годов для моделирования")

y_model_dict = {model_years[0]: pre_curves[model_years[0]]["y"].copy()}

for year in model_years[:-1]:
    next_year = year + 1
    if next_year in pre_curves:
        y_model_dict[next_year] = one_model_step_phi_star(y_model_dict[year])

# ==========================================================
# 10. ФУНКЦИОНАЛЫ ДЛЯ phi*
# ==========================================================

def one_step_functional():
    J = 0.0
    n = 0

    for year0, year1 in all_transition_years:
        y0 = pre_curves[year0]["y"]
        y1 = pre_curves[year1]["y"]

        shares_now = np.diff(y0)
        gamma_now = shares_now / DX

        phi_vals = phi_star_func(gamma_now)
        psi_vals = psi0(gamma_now)

        q_vals = (1.0 - phi_vals) * psi_vals
        A_vals = DX * np.cumsum(q_vals)
        B_val = DX * np.sum(q_vals)

        rhs = (R_RATE / (1.0 - BETA_MODEL)) * (A_vals - y0[1:] * B_val)
        y_next_model = y0[1:] + rhs

        J += np.sum((y_next_model - y1[1:]) ** 2)
        n += len(y1[1:])

    return J, J / n, np.sqrt(J / n)

def trajectory_functional():
    J = 0.0
    n = 0

    for year in sorted(y_model_dict.keys()):
        if year in pre_curves:
            y_model = y_model_dict[year]
            y_obs = pre_curves[year]["y"]

            J += np.sum((y_model[1:] - y_obs[1:]) ** 2)/10
            n += len(y_obs[1:])

    return J, J / n, np.sqrt(J / n)

J_one, MSE_one, RMSE_one = one_step_functional()
J_traj, MSE_traj, RMSE_traj = trajectory_functional()

print("\n==============================")
print("ФУНКЦИОНАЛЫ ДЛЯ phi*")
print("==============================")
print(f"One-step J    = {J_one:.8e}")
print(f"One-step MSE  = {MSE_one:.8e}")
print(f"One-step RMSE = {RMSE_one:.8e}")
print()
print(f"Trajectory J    = {J_traj:.8e}")
print(f"Trajectory MSE  = {MSE_traj:.8e}")
print(f"Trajectory RMSE = {RMSE_traj:.8e}")

# ==========================================================
# 11. ОШИБКА ПО ГОДАМ
# ==========================================================

years_error = []
errors_year = []

for year in sorted(y_model_dict.keys()):
    if year in pre_curves:
        err = np.sum((y_model_dict[year][1:] - pre_curves[year]["y"][1:]) ** 2)
        years_error.append(year)
        errors_year.append(err)

plt.figure(figsize=(10, 5))
plt.plot(years_error, errors_year, marker="o", linewidth=2)
plt.title("Ошибка модели по годам")
plt.xlabel("Год")
plt.ylabel("Сумма квадратов ошибок")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()