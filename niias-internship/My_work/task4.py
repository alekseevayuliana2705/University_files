import pandas as pd
from sqlalchemy import create_engine
from datetime import timedelta


def main():
    # 1. Подключение к БД и загрузка данных
    engine = create_engine("postgresql://postgres:1234@localhost:5432/postgres")

    query = """
            SELECT w.dor_kod, \
                   w.id_z, \
                   p.pereg_ms_id, \
                   p.stan1_id, \
                   p.stan2_id, \
                   w.dt_nd_p AS window_start, \
                   w.dt_kd_p AS window_end
            FROM icd4.wnd_z w
                     JOIN icd4.wnd_z_place p
                          ON w.dor_kod = p.dor_kod AND w.id_z = p.id_z
            WHERE w.status_fact = 2
              AND w.flg_wnd = 1
              AND p.tip = 1
              AND w.dt_kd_p > w.dt_nd_p
            ORDER BY p.pereg_ms_id, w.dor_kod, w.id_z, w.dt_nd_p \
            """

    df = pd.read_sql(query, engine)
    print(f"Загружено {len(df)} окон")

    # 2. Функция для объединения окон по правилам
    def merge_group_windows(group_df):
        if len(group_df) == 0:
            return pd.DataFrame()

        # Сортируем по времени начала
        windows = group_df.sort_values('window_start')
        merged_windows = []

        # Первое окно в группе
        current_start = windows.iloc[0]['window_start']
        current_end = windows.iloc[0]['window_end']
        count = 1

        for i in range(1, len(windows)):
            next_start = windows.iloc[i]['window_start']
            next_end = windows.iloc[i]['window_end']

            # Правило 1: Полное вложение - пропускаем
            if next_start >= current_start and next_end <= current_end:
                count += 1
                continue

            # Правило 2: Пересекающиеся или смежные окна
            if next_start <= current_end:
                current_end = max(current_end, next_end)
                count += 1
            else:
                # Сохраняем текущее объединенное окно
                duration = (current_end - current_start).total_seconds()
                merged_windows.append({
                    **windows.iloc[0][['pereg_ms_id', 'dor_kod', 'id_z', 'stan1_id', 'stan2_id']].to_dict(),
                    'window_start': current_start,
                    'window_end': current_end,
                    'duration': duration,
                    'count_windows': count
                })
                # Начинаем новое окно
                current_start, current_end = next_start, next_end
                count = 1

        # Добавляем последнее окно
        duration = (current_end - current_start).total_seconds()
        merged_windows.append({
            **windows.iloc[0][['pereg_ms_id', 'dor_kod', 'id_z', 'stan1_id', 'stan2_id']].to_dict(),
            'window_start': current_start,
            'window_end': current_end,
            'duration': duration,
            'count_windows': count
        })

        return pd.DataFrame(merged_windows)

    # 3. Обработка всех групп
    print("Объединение окон...")
    result = df.groupby(['pereg_ms_id', 'dor_kod', 'id_z'], group_keys=False).apply(merge_group_windows)
    result.reset_index(drop=True, inplace=True)

    # 4. Сохранение результатов
    print("Сохранение в БД...")
    result.to_sql('merged_windows', engine, if_exists='replace', index=False)

    print(f"Готово! Получено {len(result)} объединенных интервалов")
    return result


if __name__ == "__main__":
    final_result = main()