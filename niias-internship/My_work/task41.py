import pandas as pd
from sqlalchemy import create_engine

# Подключение к базе данных
engine = create_engine("postgresql://postgres:1234@localhost:5432/postgres")

# 1. Получаем данные из базы с фильтрацией
query = """
        SELECT z.dor_kod, \
               z.id_z, \
               z.dt_nd_p AS window_start, \
               z.dt_kd_p AS window_end, \
               p.pereg_ms_id, \
               p.stan1_id, \
               p.stan2_id
        FROM icd4.wnd_z z \
                 JOIN \
             icd4.wnd_z_place p ON p.dor_kod = z.dor_kod AND p.id_z = z.id_z
        WHERE z.status_fact = 2
          AND z.flg_wnd = 1
          AND p.tip = 1
          AND z.dt_kd_p > z.dt_nd_p -- Положительная продолжительность
          AND p.pereg_ms_id <> 0
        ORDER BY p.pereg_ms_id, z.dt_nd_p \
        """

df = pd.read_sql(query, engine)


# 2. Функция для объединения интервалов на перегоне
def merge_windows_on_peregon(group_df):
    if len(group_df) == 0:
        return pd.DataFrame()

    # Получаем идентификатор перегона
    pereg_id = group_df.name

    # Сортируем окна по времени начала
    windows = group_df.sort_values('window_start').to_dict('records')
    merged = []
    current = None

    for window in windows:
        if current is None:
            # Первое окно в группе
            current = {
                'dor_kod': window['dor_kod'],
                'pereg_ms_id': pereg_id,
                'stan1_id': window['stan1_id'],
                'stan2_id': window['stan2_id'],
                'window_start': window['window_start'],
                'window_end': window['window_end'],
                'window_ids': [window['id_z']],
                'windows_count': 1
            }
        else:
            # Проверяем условия объединения
            if window['window_start'] <= current['window_end']:
                # Окна пересекаются или соприкасаются
                new_start = min(current['window_start'], window['window_start'])
                new_end = max(current['window_end'], window['window_end'])

                # Обновляем текущий интервал
                current['window_start'] = new_start
                current['window_end'] = new_end
                current['window_ids'].append(window['id_z'])
                current['windows_count'] += 1
            else:
                # Нельзя объединить - сохраняем текущий и начинаем новый
                merged.append(current)
                current = {
                    'dor_kod': window['dor_kod'],
                    'pereg_ms_id': pereg_id,
                    'stan1_id': window['stan1_id'],
                    'stan2_id': window['stan2_id'],
                    'window_start': window['window_start'],
                    'window_end': window['window_end'],
                    'window_ids': [window['id_z']],
                    'windows_count': 1
                }

    if current is not None:
        merged.append(current)

    return pd.DataFrame(merged)


# 3. Обрабатываем каждый перегон (исправленное группирование)
result = (
    df.groupby('pereg_ms_id', group_keys=False, as_index=False)
    .apply(lambda g: merge_windows_on_peregon(g))
    .reset_index(drop=True)
)

# 4. Добавляем продолжительность в минутах
result['duration_minutes'] = (result['window_end'] - result['window_start']).dt.total_seconds() / 60

# 5. Сохраняем результат
result.to_sql(
    'merged_windows_result',
    engine,
    schema='public',
    if_exists='replace',
    index=False
)

print(f"Готово! Обработано {len(result)} объединенных интервалов")
print("Пример результата:")
print(result.head(3))

# Закрываем соединение
engine.dispose()