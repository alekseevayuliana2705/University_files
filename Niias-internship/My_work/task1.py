import pandas as pd
from sqlalchemy import create_engine, text
from datetime import timedelta

def hours_to_hms(hours):
    td = timedelta(hours=hours)
    total_seconds = int(td.total_seconds())
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

engine = create_engine("postgresql://postgres:1234@localhost:5432/postgres")

query = """
SELECT
    t.dor_kod,
    t.dt_nd,
    t.dt_kd,
    (t.dt_kd - t.dt_nd) AS duration,
    z.status_pl
FROM
    icd4.wnd_time t
JOIN icd4.wnd_z z 
    ON t.dor_kod = z.dor_kod AND t.id_z = z.id_z
WHERE t.dt_nd BETWEEN '2014-01-01' AND '2014-12-31'
	AND z.status_pl = 2
    AND t.dt_nd <= t.dt_kd
    AND EXTRACT(EPOCH FROM (t.dt_kd - t.dt_nd)) < 24*3600
ORDER BY
	t.dor_kod
"""
df = pd.read_sql(query, engine)
n = 3

# ------------------------------------- 1 PART -----------------------------------
# Подсчет среднего арифметического
df['duration_minutes'] = df['duration'].dt.total_seconds() / 60
arith_mean_min = df['duration_minutes'].mean() # в минутах
arith_mean = abs(df['duration']).mean() # days hours:minutes:seconds
print(f'Средняя продолжительность окон в минутах: {round(arith_mean_min/60, n)}')
print(f'Средняя продолжительность окон: {timedelta(seconds=int(arith_mean.total_seconds()))}')

# Подсчет среднего квадратического отклонения
std_biased = df['duration_minutes'].std(ddof=0) # Смещенное
std_deviation = df['duration_minutes'].std(ddof=1) # Несмещенное
print(f'СКО (смещенное) в часах: {hours_to_hms(std_biased/60)}')
print(f'СКО (несмещенное) в часах: {hours_to_hms(std_deviation/60)}')

# Подсчет оценок дисперсий
var_biased = df['duration_minutes'].var(ddof=0) # Смещенная
var_deviation = df['duration_minutes'].var(ddof=1) # Несмещенная
print(f'Дисперсия (смещенная) в часах: {hours_to_hms(var_biased/3600)}')
print(f'Дисперсия (несмещенная) в часах: {hours_to_hms(var_deviation/3600)}')

# ------------------------------------- 2 PART -----------------------------------
df['month'] = df['dt_nd'].dt.month
monthly_stats = df.groupby('month').agg(
    # Среднее арифметическое
    mean=('duration_minutes', 'mean'),
    # СКО (смещенное и несмещенное)
    std_bia_month=('duration_minutes', lambda x: x.std(ddof=0)),
    std_dev_month=('duration_minutes', lambda x: x.std(ddof=1)),
    # Дисперсия (смещенная и несмещенная)
    var_bia_month=('duration_minutes', lambda x: x.var(ddof=0)),
    var_dev_month=('duration_minutes', lambda x: x.var(ddof=1))
).reset_index()

# Создаем таблицу в БД
monthly_stats.to_sql('monthly_stats_temp', engine, if_exists='replace', index=False)
create_table_sql = text("""
CREATE TABLE IF NOT EXISTS public.monthly_stats AS
SELECT 
    month,
    ROUND(mean::numeric/60, 2) AS avg_duration_minutes,
    ROUND(std_bia_month::numeric/60, 2) AS std_bia,
    ROUND(std_dev_month::numeric/60, 2) AS std_dev,
    ROUND(var_bia_month::numeric/3600, 2) AS var_bia,
    ROUND(var_dev_month::numeric/3600, 2) AS var_dev
FROM monthly_stats_temp;
""")
with engine.begin() as conn:
    conn.execute(create_table_sql)
    conn.execute(text("DROP TABLE IF EXISTS monthly_stats_temp"))

