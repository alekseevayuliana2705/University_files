import pandas as pd
from sqlalchemy import create_engine
from ast import literal_eval

# Подключение к базе данных
engine = create_engine("postgresql://postgres:1234@localhost:5432/postgres")

# 1. Загружаем уже готовую таблицу с объединенными окнами
query = "SELECT * FROM public.merged_windows_result"
result = pd.read_sql(query, engine)

# 2. Функция для очистки дубликатов в window_ids
def clean_window_ids(row):
    try:
        # Если window_ids уже список
        if isinstance(row['window_ids'], list):
            unique_ids = list(set(row['window_ids']))  # Удаляем дубликаты
            return {
                **row.to_dict(),
                'window_ids': unique_ids,
                'windows_count': len(unique_ids)  # Пересчитываем количество
            }
        # Если window_ids строка (например, "[1,2,3]")
        elif isinstance(row['window_ids'], str):
            ids_list = literal_eval(row['window_ids'])
            unique_ids = list(set(ids_list))
            return {
                **row.to_dict(),
                'window_ids': unique_ids,
                'windows_count': len(unique_ids)
            }
    except:
        return row  # В случае ошибки оставляем как есть

# 3. Применяем функцию ко всем строкам
cleaned_result = result.apply(clean_window_ids, axis=1, result_type='expand')

# 4. Сохраняем исправленный результат
cleaned_result.to_sql(
    'cleaned_merged_windows',
    engine,
    schema='public',
    if_exists='replace',
    index=False
)

print(f"Очищено дубликатов в {len(result)} записях")
print("Пример до очистки:")
print(result[['window_ids', 'windows_count']].head(3))
print("\nПример после очистки:")
print(cleaned_result[['window_ids', 'windows_count']].head(3))

# Закрываем соединение
engine.dispose()