import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://postgres:1234@localhost:5432/postgres")

query = """
SELECT 
    dor_kod,
    id_z,
    (flg_2ab = 1) AS need_for_2ab,
    (flg_voltage = 1) AS relieve_voltage,
    (flg_wnd = 1) AS with_wnd,
    (flg_scb = 1) AS scb_required,
    (status_fact = 2) AS approved,
    ((status_pl = 1 OR status_pl = 2) AND (status_fact <> 2)) AS not_approved,
    ((status_pl = 0 OR status_pl = 3) AND (status_fact <> 2)) AS rejected,
    seq_voltage AS voltage_sequence,
    (flg_voltage > 0 AND time_voltage = 0) AS voltage_early,
    (flg_voltage > 0 AND time_voltage = 1) AS voltage_late,
    EXTRACT(EPOCH FROM (dt_kd_p - dt_nd_p))/60 AS window_duration,
    status_pl AS plan_status,
    status_fact AS fact_status,
    CASE WHEN flg_voltage > 0 THEN time_voltage ELSE 0 END AS voltage_time,
    CASE
        WHEN seq_voltage = (EXTRACT(EPOCH FROM (dt_kd_p - dt_nd_p))/60) THEN 0
        WHEN seq_voltage > (EXTRACT(EPOCH FROM (dt_kd_p - dt_nd_p))/60) THEN 1
        WHEN seq_voltage < (EXTRACT(EPOCH FROM (dt_kd_p - dt_nd_p))/60) THEN -1
        ELSE NULL
    END AS voltage_duration_comparison,
    br_count AS planned_teams,
    people_count AS planned_workers,
    br_count_fact AS actual_teams,
    people_count_fact AS actual_workers,
    (br_count_fact - br_count) AS teams_difference,
    (people_count_fact - people_count) AS workers_difference
FROM icd4.wnd_z
"""

try:
    df = pd.read_sql(query, engine)
    df.to_csv('/Users/ulianaalekseeva/Desktop/my_prob.csv', index=False)
    print("Данные успешно сохранены в CSV!")
except Exception as e:
    print(f"Ошибка при выполнении запроса: {e}")
finally:
    engine.dispose()