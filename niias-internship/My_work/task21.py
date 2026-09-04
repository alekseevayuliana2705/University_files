import pandas as pd
from sqlalchemy import create_engine
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots

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

# 2. Диаграмма для заданий 2-4 (процентное распределение)
from matplotlib.colors import LinearSegmentedColormap

colors1 = ["#D1FFC8", "#00C71D"]
colors2 = ["#FFE200", "#FF7100"]
colors3 = ["#FF8200", "#FF0900"]
cmap1 = LinearSegmentedColormap.from_list("custom_gradient", colors1)
cmap2 = LinearSegmentedColormap.from_list("custom_gradient", colors2)
cmap3 = LinearSegmentedColormap.from_list("custom_gradient", colors3)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                   vertical_spacing=0.1,
                   subplot_titles=('Общее количество окон', 'Процентное распределение по смещениям'))
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
plt.subplots_adjust(top=0.85, wspace=0.3)
metrics = [
    ('delayed_less_1h_pct', '<1 часа', cmap1),
    ('delayed_1_3h_pct', '1-3 часов', cmap2),
    ('delayed_over_3h_pct', '>3 часов', cmap3)
]
month_names = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
              'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
for i, (col, title, cmap) in enumerate(metrics):
    pivot_data = df2.pivot_table(
        index='month',
        columns='year',
        values=col,
        aggfunc='mean'
    )
    heatmap = sns.heatmap(
        pivot_data,
        ax=axes[i],
        cmap=cmap,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        vmin=0,
        vmax=100,
        cbar=False,
        xticklabels=[str(int(float(x))) for x in pivot_data.columns]
    )
    cbar = fig.colorbar(heatmap.get_children()[0], ax=axes[i], fraction=0.046, pad=0.04)
    axes[i].set_title(title, fontsize=14, pad=15)
    axes[i].set_xlabel('Год', fontsize=12)
    axes[i].set_ylabel('Месяц', fontsize=12)
    axes[i].set_yticks(range(12))
    axes[i].set_yticklabels(month_names, rotation=0)
plt.suptitle('Анализ задержек исполнения "окон" по месяцам и годам',
            fontsize=18, y=0.975)
plt.tight_layout()
plt.show()