import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# Подключение к БД
engine = create_engine("postgresql://postgres:1234@localhost:5432/postgres")

# Запрос
query = """
WITH filtered_failures AS (
    SELECT 
        COUNT(*) AS total_count,
        COUNT(CASE WHEN wz.id_reject_fact = 168 THEN 1 END) AS executor_count
    FROM icd4.wnd_z wz
    WHERE wz.status_fact <> 2
      AND wz.id_reject_fact IN (16, 37, 38, 39, 40, 99, 140, 161, 164, 165, 166, 167, 168, 169, 187)
      AND wz.dt_nd_p IS NOT NULL
      AND wz.dt_kd_p IS NOT NULL
      AND wz.dt_nd_p <= wz.dt_kd_p
)
SELECT 
    executor_count AS "Срывы из-за исполнителя",
    total_count AS "Всего срывов (выбранные причины)",
    ROUND(executor_count * 100.0 / NULLIF(total_count, 0), 2) AS "Процент"
FROM filtered_failures;
"""

df = pd.read_sql(query, engine)

# Визуализация
plt.figure(figsize=(8, 4))
plt.bar(['Срывы из-за исполнителя', 'Другие выбранные причины'],
        [df.iloc[0]['Срывы из-за исполнителя'],
        df.iloc[0]['Всего срывов (выбранные причины)'] - df.iloc[0]['Срывы из-за исполнителя']],
        color=['#ff9999', '#66b3ff'])

plt.title(f'Срывы технологических окон\n({df.iloc[0]["Процент"]}% — отказ исполнителя)')
plt.ylabel('Количество случаев')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()