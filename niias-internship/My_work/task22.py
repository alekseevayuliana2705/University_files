import pandas as pd
from sqlalchemy import create_engine
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

engine = create_engine("postgresql://postgres:1234@localhost:5432/postgres")

query_part1 = """
WITH nmb_wnd_month AS (
    SELECT 
        EXTRACT(YEAR FROM t.dt_nd) AS year,
        EXTRACT(MONTH FROM t.dt_nd) AS month,
        COUNT(*) AS total_compl_wnd
    FROM 
        icd4.wnd_z z
    JOIN 
        icd4.wnd_time t ON z.dor_kod = t.dor_kod AND z.id_z = t.id_z
    WHERE 
        z.status_fact = 2
        AND t.dt_nd BETWEEN '2005-01-01' AND '2020-12-31'
    GROUP BY 
        year,
        month
)
SELECT 
    year,
    month,
    total_compl_wnd
FROM 
    nmb_wnd_month
ORDER BY 
    year, month;
"""

query_part2 = """
WITH wnd_st AS (
    SELECT
        EXTRACT(YEAR FROM t.dt_nd) AS year,
        EXTRACT(MONTH FROM t.dt_nd) AS month,
        COUNT(*) FILTER (
            WHERE z.status_fact = 2
            AND t.dt_nd IS NOT NULL 
            AND z.dt_nd_p IS NOT NULL
        ) AS total_wnd,
        COUNT(*) FILTER (
            WHERE z.status_fact = 2 
            AND t.dt_nd IS NOT NULL 
            AND z.dt_nd_p IS NOT NULL
            AND ABS(EXTRACT(EPOCH FROM (t.dt_nd - z.dt_nd_p))) < 3600
        ) AS delayed_less_1h,
        COUNT(*) FILTER (
            WHERE z.status_fact = 2 
            AND t.dt_nd IS NOT NULL 
            AND z.dt_nd_p IS NOT NULL
            AND ABS(EXTRACT(EPOCH FROM (t.dt_nd - z.dt_nd_p))) BETWEEN 3600 AND 10800
        ) AS delayed_1_3h,
        COUNT(*) FILTER (
            WHERE z.status_fact = 2 
            AND t.dt_nd IS NOT NULL 
            AND z.dt_nd_p IS NOT NULL
            AND ABS(EXTRACT(EPOCH FROM (t.dt_nd - z.dt_nd_p))) > 10800
        ) AS delayed_over_3h
    FROM
        icd4.wnd_z z
    JOIN
        icd4.wnd_time t ON z.dor_kod = t.dor_kod AND z.id_z = t.id_z
    WHERE
        t.dt_nd BETWEEN '2005-01-01' AND '2020-12-31'
        AND z.status_fact = 2
    GROUP BY
        year, month
)
SELECT
    year,
    month,
    total_wnd,
    delayed_less_1h,
    delayed_1_3h,
    delayed_over_3h,
    ROUND(delayed_less_1h * 100.0 / NULLIF(total_wnd, 0), 2) AS delayed_less_1h_pct,
    ROUND(delayed_1_3h * 100.0 / NULLIF(total_wnd, 0), 2) AS delayed_1_3h_pct,
    ROUND(delayed_over_3h * 100.0 / NULLIF(total_wnd, 0), 2) AS delayed_over_3h_pct
FROM
    wnd_st
ORDER BY
    year, month;
"""

df1 = pd.read_sql(query_part1, engine)
df2 = pd.read_sql(query_part2, engine)
def create_date(row):
    return pd.to_datetime(f"{int(row['year'])}-{int(row['month'])}-01")
df1['date'] = df1.apply(create_date, axis=1)
df2['date'] = df2.apply(create_date, axis=1)

# 1. Диаграмма для задания 1
plt.figure(figsize=(14, 7))
plt.plot(df1['date'], df1['total_compl_wnd'], label='Всего исполненных окон', color='blue', linewidth=2)
plt.title('Общее количество исполненных окон по месяцам (2005-2020)')
plt.xlabel('Год')
plt.ylabel('Количество окон')
plt.grid(True)
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Диаграмма для заданий 2-4
df2['date_str'] = df2['date'].dt.strftime('%Y-%m')
plt.figure(figsize=(16, 8))
width = 0.7
x = np.arange(len(df2))
bottom = np.zeros(len(df2))
colors = ['#2ecc71', '#f39c12', '#e74c3c']
labels = ['<1 часа', '1-3 часа', '>3 часов']
for i, col in enumerate(['delayed_less_1h_pct', 'delayed_1_3h_pct', 'delayed_over_3h_pct']):
    plt.bar(x, df2[col], width, bottom=bottom, color=colors[i], label=labels[i])
    bottom += df2[col]

plt.title('Анализ задержек исполнения "окон" по месяцам и годам', fontsize=16)
plt.xlabel('Месяц и год', fontsize=12)
plt.ylabel('Процент окон', fontsize=12)
plt.ylim(0, 100)
plt.xticks(x[::12], df2['date_str'][::12], rotation=45)
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()