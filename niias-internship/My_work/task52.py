import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from matplotlib.ticker import PercentFormatter

# Подключение к БД
engine = create_engine("postgresql://postgres:1234@localhost:5432/postgres")

# SQL-запрос (без изменений)
query = """
        WITH failed_windows AS (SELECT wz.id_z, \
                                       rt.id_reason_type, \
                                       rt.name AS reason_type_name \
                                FROM icd4.wnd_z wz \
                                         JOIN icd4.reason_wnd rw ON wz.id_reject_fact = rw.id_reason \
                                         JOIN icd4.reason_group rg ON rw.id_reason = rg.id_reason \
                                         JOIN icd4.reason_type rt ON rg.id_reason_type = rt.id_reason_type \
                                WHERE wz.status_fact <> 2 \
                                  AND wz.dt_nd_p IS NOT NULL \
                                  AND wz.dt_kd_p IS NOT NULL \
                                  AND wz.dt_nd_p <= wz.dt_kd_p \
                                  AND rt.id_reason_type IN (1, 4, 6, 10, 11, 12, 13, 14)),
             total_failed AS (SELECT COUNT(*) AS total_count \
                              FROM failed_windows),
             reason_stats AS (SELECT reason_type_name, \
                                     COUNT(*)                               AS type_count, \
                                     (SELECT total_count FROM total_failed) AS total_count \
                              FROM failed_windows \
                              GROUP BY reason_type_name)
        SELECT reason_type_name                           AS "Тип причины", \
               type_count                                 AS "Количество", \
               ROUND(type_count * 100.0 / total_count, 2) AS "Процент", \
               RANK()                                        OVER (ORDER BY type_count DESC) AS "Ранг"
        FROM reason_stats
        ORDER BY type_count DESC; \
        """

df = pd.read_sql(query, engine)

# Проверяем, есть ли данные
if df.empty:
    print("Нет данных для отображения. Проверьте наличие невыполненных окон.")
else:
    # Создаем фигуру
    plt.figure(figsize=(12, 6))

    # Цветовая палитра
    colors = plt.cm.tab20.colors[:len(df)]

    # Горизонтальная бар-диаграмма
    bars = plt.barh(
        df['Тип причины'],
        df['Процент'],
        color=colors,
        height=0.6
    )

    # Настройки оформления
    plt.title('Анализ причин невыполнения технологических окон',
              pad=20, fontsize=14)
    plt.xlabel('Доля от общего числа невыполненных окон, %', fontsize=12)
    plt.ylabel('Тип причины', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    # Добавляем значения
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(
            width + 0.5,  # Смещение от столбца
            bar.get_y() + bar.get_height() / 2,  # По центру столбца
            f"{width:.2f}% ({df.iloc[i]['Количество']})",  # Формат: "процент (n=число)"
            va='center',
            fontsize=10
        )

    # Убираем лишние отступы
    plt.tight_layout()

    # Сохраняем и показываем график
    plt.savefig('distribution_of_reasons.png', dpi=120, bbox_inches='tight')
    plt.show()