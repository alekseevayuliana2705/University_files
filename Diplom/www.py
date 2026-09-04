# =============================================================================
# INVERSE PROBLEM: INCOME DISTRIBUTION DYNAMICS
# TRAINING: 1980–2010
# STABILITY TEST OF φ: 2011–2016
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.optimize import minimize
import os
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# PARAMETERS
# =============================================================================

DATA_PATH = "/Users/ulianaalekseeva/Downloads/Germany/"
TRAIN_YEARS = range(1980, 2011)
TEST_YEARS  = range(2011, 2017)
r = 0.03

DECILES = [
    "p0p10","p10p20","p20p30","p30p40","p40p50",
    "p50p60","p60p70","p70p80","p80p90","p90p100"
]

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_for_years(years):

    files = {
        "post_0_50": "Post_0_50.csv",
        "post_50_100": "Post_50_100.csv",
        "pre_0_50":  "Pre_0_50.csv",
        "pre_50_100":"Pre_50_100.csv",
    }

    data = {
        k: pd.read_csv(os.path.join(DATA_PATH, v),
                       sep=";", skiprows=5, header=None)
        for k, v in files.items()
    }

    def sf(x):
        try:
            return float(str(x).replace(",", "."))
        except:
            return None

    post, pre = {}, {}

    for key in ["post_0_50","post_50_100"]:
        for _, row in data[key].iterrows():
            y = sf(row[1])
            if y in years:
                post.setdefault(y, {})[row[0].strip()] = sf(row[2])

    for key in ["pre_0_50","pre_50_100"]:
        for _, row in data[key].iterrows():
            y = sf(row[1])
            if y in years:
                pre.setdefault(y, {})[row[0].strip()] = sf(row[2])

    return post, pre

# =============================================================================
# LORENZ CURVES
# =============================================================================

def build_lorenz(data):

    out = {}
    for y, d in data.items():
        if all(k in d for k in DECILES):

            s = np.array([d[k] for k in DECILES])
            s /= s.sum()

            out[y] = {
                "x": np.linspace(0,1,11),
                "y": np.concatenate([[0], np.cumsum(s)])
            }
    return out


def make_splines(lorenz):

    spl = {}
    for y, d in lorenz.items():

        sp = PchipInterpolator(d["x"], d["y"])
        xd = np.linspace(0,1,400)
        yd = sp(xd)
        gd = np.maximum(np.gradient(yd, xd), 1e-8)

        spl[y] = {"s": sp, "x": xd, "y": yd, "g": gd}

    return spl

# =============================================================================
# ψ₀(γ)
# =============================================================================

def build_psi0(post, pre):

    years = sorted(set(post) & set(pre))
    gp, gq = [], []

    for y in years:
        for k in DECILES:
            if k in post[y] and k in pre[y]:
                gp.append(pre[y][k])
                gq.append(post[y][k])

    gp, gq = np.array(gp), np.array(gq)
    gp /= gp.sum()
    gq /= gq.sum()

    f = interp1d(gp, gq/gp, fill_value="extrapolate")

    return lambda g: g * f(np.clip(g, gp.min(), gp.max()))

# =============================================================================
# φ(γ) — bounded logistic
# =============================================================================

def phi_logistic(g, beta, phi0, phi1, k, g0):
    return phi1 + (phi0 - phi1)/(1 + np.exp(k*(g - g0)))

# =============================================================================
# G(x; φ)
# =============================================================================

def compute_G(x, spl, phi, psi0):

    xg, g = spl["x"], spl["g"]
    f = (1 - phi(g)) * psi0(g)

    i = np.searchsorted(xg, x)
    I1 = 0 if i == 0 else np.trapz(f[:i], xg[:i])
    I2 = np.trapz(f, xg)

    return I1 - spl["s"](x) * I2

# =============================================================================
# TRAINING φ ON 1980–2010 (φ ≤ 1)
# =============================================================================

def train_phi(lorenz, psi0):

    spl = make_splines(lorenz)
    years = sorted(spl)

    def J(p):

        beta, phi0, phi1, k, g0 = p

        if not (0 < beta < 1 and 0 < phi1 <= phi0 <= 1 and k > 0):
            return 1e12

        phi = lambda g: phi_logistic(g, beta, phi0, phi1, k, g0)
        err = 0

        for y0, y1 in zip(years[:-1], years[1:]):

            S0, S1 = spl[y0], spl[y1]
            G = np.array([compute_G(x, S0, phi, psi0)
                          for x in S0["x"]])

            dy = r/(1-beta) * G
            err += np.trapz((S1["y"] - S0["y"] - dy)**2, S0["x"])

        return err

    x0 = [0.45, 0.95, 0.7, 10, 0.5]
    b  = [(0.1,0.9),(0.5,1),(0.1,0.9),(1,50),(0.1,0.9)]

    res = minimize(J, x0, bounds=b, method="L-BFGS-B")

    beta = res.x[0]
    phi  = lambda g: phi_logistic(g, beta, *res.x[1:])

    return beta, phi, res

# =============================================================================
# ONE-STEP φ (β fixed)
# =============================================================================

def train_phi_one_step(S0, S1, psi0, beta):

    def J(p):

        phi0, phi1, k, g0 = p

        if not (0 < phi1 <= phi0 <= 1 and k > 0):
            return 1e10

        phi = lambda g: phi_logistic(g, beta, phi0, phi1, k, g0)
        G = np.array([compute_G(x, S0, phi, psi0)
                      for x in S0["x"]])

        dy = r/(1-beta) * G
        return np.trapz((S1["y"] - S0["y"] - dy)**2, S0["x"])

    x0 = [0.95, 0.7, 10, 0.5]
    b  = [(0.5,1),(0.1,0.9),(1,50),(0.1,0.9)]

    return minimize(J, x0, bounds=b, method="L-BFGS-B")

# =============================================================================
# MAIN
# =============================================================================

post_tr, pre_tr = load_data_for_years(TRAIN_YEARS)
post_te, pre_te = load_data_for_years(TEST_YEARS)

L_tr = build_lorenz(post_tr)
L_te = build_lorenz(post_te)

psi0 = build_psi0(post_tr, pre_tr)

beta, phi_base, _ = train_phi(L_tr, psi0)

spl_te = make_splines(L_te)
years = sorted(spl_te)

phis = [
    (f"{y0}→{y1}",
     train_phi_one_step(spl_te[y0], spl_te[y1], psi0, beta).x)
    for y0, y1 in zip(years[:-1], years[1:])
]

# =============================================================================
# PLOT (исправленная версия)
# =============================================================================

g = np.linspace(0,1,400)

plt.figure(figsize=(9,6))

# Базовая φ (1980–2010)
plt.plot(
    g,
    phi_base(g),
    color="black",
    linewidth=3,
    label="1980–2010"
)

# Цветовая палитра
colors = plt.cm.viridis(np.linspace(0,1,len(phis)))

for (lbl, p), c in zip(phis, colors):

    phi_vals = phi_logistic(g, beta, *p)

    plt.plot(
        g,
        phi_vals,
        linestyle="--",
        linewidth=2,
        alpha=0.85,
        color=c,
        label=lbl
    )

plt.xlabel("γ")
plt.ylabel("φ(γ)")
plt.title("Stability of φ (2011–2016)")
plt.legend(loc="best", fontsize=9)
plt.grid(alpha=0.3)

plt.savefig("phi_stability.png", dpi=300, bbox_inches="tight")
plt.show()


