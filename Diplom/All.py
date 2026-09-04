import pandas as pd
import matplotlib.pyplot as plt

FILES = [
#    "phi_star_raw_russia.csv",
#    "phi_star_raw_germany.csv",
#    "phi_star_raw_usa.csv",
#    "phi_star_raw_uk.csv",
#    "phi_star_raw_sweden.csv",
    "phi_star_raw_spain.csv",
#    "phi_star_raw_poland.csv",
#    "phi_star_raw_netherlands.csv",
#    "phi_star_raw_france.csv",
    "phi_star_raw_china.csv",
    "phi_star_raw_japan.csv",
#    "phi_star_raw_italy.csv",
#    "phi_star_raw_australia.csv",
    "phi_star_raw_kazahstan.csv",
#    "phi_star_raw_brazil.csv"

]

plt.figure(figsize=(10, 6))

for file in FILES:
    df = pd.read_csv(file)

    country = df["country"].iloc[0]

    plt.plot(
        df["gamma"],
        df["phi_star_raw"],
        linewidth=3,
        label=country
    )

plt.title(r"Сравнение функций $\phi^*(\gamma)$ по странам")
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$\phi^*(\gamma)$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()