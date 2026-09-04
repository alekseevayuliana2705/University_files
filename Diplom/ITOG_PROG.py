# ==========================================================
# ESTIMATION OF φ(γ) WITH TAX REDISTRIBUTION ψ
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
import os

# ==========================================================
# USER PARAMETERS
# ==========================================================

DATA_PATH = "/Users/ulianaalekseeva/Downloads/Germany/"

PERIODS = [
    (1980, 2000),
    (1980, 2000)
]

r = 0.03
beta = 0.95
w = 1.0

DECILES = [
"p0p10","p10p20","p20p30","p30p40","p40p50",
"p50p60","p60p70","p70p80","p80p90","p90p100"
]

# ==========================================================
# LOAD DATA
# ==========================================================

def load_file(file):

    df = pd.read_csv(
        file,
        sep=";",
        skiprows=5,
        header=None
    )

    df.columns = ["decile","year","income"]

    df["decile"] = df["decile"].str.strip()

    df["year"] = pd.to_numeric(df["year"],errors="coerce")
    df["income"] = pd.to_numeric(
        df["income"].astype(str).str.replace(",","."),
        errors="coerce"
    )

    return df.dropna()


def load_data():

    pre_low = load_file(os.path.join(DATA_PATH,"Pre_0_50.csv"))
    pre_high = load_file(os.path.join(DATA_PATH,"Pre_50_100.csv"))

    post_low = load_file(os.path.join(DATA_PATH,"Post_0_50.csv"))
    post_high = load_file(os.path.join(DATA_PATH,"Post_50_100.csv"))

    pre = pd.concat([pre_low,pre_high])
    post = pd.concat([post_low,post_high])

    return pre,post


# ==========================================================
# BUILD LORENZ CURVES
# ==========================================================

def build_lorenz(df):

    years = sorted(df.year.unique())

    lorenz = {}
    
    for y in years:
        d = df[df.year==y]

        if len(d)==10:

            inc = d.sort_values("decile").income.values

            shares = inc/inc.sum()

            lorenz[y] = {
                "x":np.linspace(0,1,11),
                "y":np.concatenate([[0],np.cumsum(shares)])
            }
    
    return lorenz


# ==========================================================
# SPLINES
# ==========================================================

def make_splines(lorenz):

    spl = {}

    for y,d in lorenz.items():

        sp = PchipInterpolator(d["x"],d["y"])

        xd = np.linspace(0,1,400)
        yd = sp(xd)

        g = np.maximum(np.gradient(yd,xd),1e-6)

        spl[y] = {"g":g}

    return spl


# ==========================================================
# BUILD TAX OPERATOR ψ
# ==========================================================

def build_tax_function(pre,post,start,end):

    pre = pre[(pre.year>=start)&(pre.year<=end)]
    post = post[(post.year>=start)&(post.year<=end)]

    gp=[]
    gq=[]

    for y in sorted(pre.year.unique()):

        d0 = pre[pre.year==y].sort_values("decile")
        d1 = post[post.year==y].sort_values("decile")

        if len(d0)==10 and len(d1)==10:

            gp.extend(d0.income.values)
            gq.extend(d1.income.values)

    gp=np.array(gp)
    gq=np.array(gq)

    gp/=gp.sum()
    gq/=gq.sum()

    order=np.argsort(gp)

    gp=gp[order]
    gq=gq[order]

    gp,idx=np.unique(gp,return_index=True)
    gq=gq[idx]

    return PchipInterpolator(gp,gq)


# ==========================================================
# DIFFERENTIAL EQUATION
# ==========================================================

def phi_from_equation(k,kdot):

    num = r*k + w - kdot
    den = k + w/r

    return (beta*r + (1-beta)*(num/den))/r


# ==========================================================
# COLLECT DATA
# ==========================================================

def collect_phi_points(spl,psi,start,end):

    years = [y for y in spl if start<=y<=end]
    years = sorted(years)

    gamma=[]
    phi=[]

    for y0,y1 in zip(years[:-1],years[1:]):

        g0 = spl[y0]["g"]
        g1 = spl[y1]["g"]

        kdot = g1-g0

        gamma_tax = psi(g0)

        phi_data = phi_from_equation(gamma_tax,kdot)

        mask = np.isfinite(phi_data)

        gamma.extend(gamma_tax[mask])
        phi.extend(phi_data[mask])

    gamma=np.array(gamma)
    phi=np.array(phi)

    return gamma,phi


# ==========================================================
# PARAMETRIC φ
# ==========================================================

def phi_param(g,a,b,c,d):

    return b + (a-b)/(1+np.exp(c*(g-d)))


def estimate_phi(gamma,phi):

    def loss(theta):

        return np.mean((phi_param(gamma,*theta)-phi)**2)

    x0=[2,0.5,10,0.5]

    bounds=[
        (0,10),
        (0,5),
        (1,50),
        (0.1,0.9)
    ]

    res=minimize(loss,x0,bounds=bounds)

    return res.x


# ==========================================================
# MAIN
# ==========================================================

pre,post = load_data()

lorenz = build_lorenz(post)

spl = make_splines(lorenz)

phi_results={}

for start,end in PERIODS:

    psi = build_tax_function(pre,post,start,end)

    gamma,phi = collect_phi_points(spl,psi,start,end)

    params = estimate_phi(gamma,phi)

    phi_results[(start,end)] = lambda g,p=params: phi_param(g,*p)

    print("Period",start,end,"parameters:",params)


# ==========================================================
# PLOT
# ==========================================================

g = np.linspace(0,1,400)

plt.figure(figsize=(9,6))

styles=["-","--","-.",":"]

for i,((s,e),phi) in enumerate(phi_results.items()):

    plt.plot(
        g,
        phi(g),
        linestyle=styles[i],
        linewidth=3,
        label=f"φ ({s}-{e})"
    )

plt.xlabel("γ")
plt.ylabel("φ(γ)")
plt.title("Estimated consumption function φ(γ)")

plt.grid(alpha=0.3)

plt.legend()

plt.show()
