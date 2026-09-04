import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap
from sqlalchemy import create_engine

# Настройки графиков
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True  # Автоподгонка размеров

# Подключение к БД
engine = create_engine("postgresql://postgres:1234@localhost:5432/postgres")

# SQL-запрос (без изменений)
query_all = """
            WITH rejected_windows AS (SELECT wz.id_reject_fact, \
                                             rw.name  AS reason_name, \
                                             COUNT(*) AS rejection_count \
                                      FROM icd4.wnd_z wz \
                                               JOIN \
                                           icd4.reason_wnd rw ON wz.id_reject_fact = rw.id_reason \
                                      WHERE wz.status_fact <> 2 \
                                        AND wz.dt_nd_p IS NOT NULL \
                                        AND wz.dt_kd_p IS NOT NULL \
                                        AND wz.dt_nd_p <= wz.dt_kd_p \
                                        AND wz.id_reject_fact IS NOT NULL \
                                        AND rw.name <> '-' \
                                        AND EXTRACT(MONTH FROM wz.dt_nd_p) IN (12, 1, 2) -- Зимние месяцы
                                      GROUP BY wz.id_reject_fact, rw.name)
            SELECT reason_name                                                                             AS "Причина невыполнения", \
                   rejection_count                                                                         AS "Количество случаев", \
                   ROUND(rejection_count * 100.0 / (SELECT SUM(rejection_count) FROM rejected_windows), \
                         2)                                                                                AS "Процент от общего числа"
            FROM rejected_windows
            ORDER BY rejection_count DESC LIMIT 5; \
            """

# Получаем данные
df_all = pd.read_sql(query_all, engine)

# Автоматический перенос длинных названий (25 символов в строке)
df_all["Причина"] = df_all["Причина невыполнения"].apply(
    lambda x: '\n'.join(wrap(x, width=25))
)

# Создаем график с увеличенной высотой для переноса строк
fig, ax = plt.subplots(figsize=(10, 6))  # Увеличена высота с 5 до 6

# Горизонтальные столбцы с цветовой палитрой
colors = plt.cm.tab10.colors  # Используем встроенную палитру
bars = ax.barh(
    df_all["Причина"],
    df_all["Количество случаев"],
    color=colors[:len(df_all)]  # Берем нужное количество цветов
)

# Добавляем подписи с количеством и процентом
for i, bar in enumerate(bars):
    width = bar.get_width()
    percent = df_all.iloc[i]["Процент от общего числа"]
    label_text = f'{width:.0f} ({percent}%)'  # Формат "число (процент)"

    ax.text(
        width + max(df_all["Количество случаев"]) * 0.01,  # Смещение от столбца
        bar.get_y() + bar.get_height() / 2,  # По центру столбца
        label_text,
        va='center',
        ha='left',
        fontsize=10
    )

# Настройка внешнего вида
ax.set_title('Топ-5 причин невыполнения технологических окон', pad=20, fontsize=14)
ax.set_xlabel('Количество случаев', fontsize=12)
ax.set_ylabel('Причины', fontsize=12)
ax.grid(axis='x', linestyle=':', alpha=0.7)

# Убираем лишние отступы
plt.tight_layout()

# Сохраняем и показываем график
plt.savefig('top_reasons_with_percent.png', dpi=120, bbox_inches='tight')
plt.show()