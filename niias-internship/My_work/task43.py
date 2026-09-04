import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import seaborn as sns

# Подключение к базе данных
engine = create_engine("postgresql://postgres:1234@localhost:5432/postgres")
df = pd.read_sql("SELECT * FROM public.cleaned_merged_windows", engine)

# Преобразование данных
df['window_start'] = pd.to_datetime(df['window_start'])
df['window_end'] = pd.to_datetime(df['window_end'])
df['hour'] = df['window_start'].dt.hour

# 1. Простой график количества окон по перегонам (топ-15)
plt.figure(figsize=(12, 6))

# Берем топ-10 записей с наибольшим windows_count
top_merged = df.nlargest(10, 'windows_count').sort_values('windows_count', ascending=True)

# Создаем понятные подписи для каждого интервала
labels = [
    f"{row['duration_minutes']/60} часов\n"
    f"Перегон {row['pereg_ms_id']}"
    for _, row in top_merged.iterrows()
]

plt.barh(
    y=labels,
    width=top_merged['windows_count'],
    color='#4c72b0',
    edgecolor='darkblue',
    alpha=0.7
)

plt.title('Топ-10 объединенных окон по количеству включенных окон', fontsize=14, pad=20)
plt.xlabel('Количество объединенных окон', fontsize=12)
plt.ylabel('Интервал и перегон', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Добавляем значения на каждый столбец
for i, value in enumerate(top_merged['windows_count']):
    plt.text(
        value + 0.2,  # Смещение от столбца
        i,            # Позиция по y
        str(value),
        va='center',
        fontsize=11
    )

plt.tight_layout()
plt.savefig('top_merged_windows.png', dpi=120, bbox_inches='tight')
plt.close()

# 2. Распределение продолжительности окон (упрощенное)
plt.figure(figsize=(10, 6))
plt.hist(df['duration_minutes'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Как долго длятся технологические окна?', fontsize=14)
plt.xlabel('Продолжительность (минуты)', fontsize=12)
plt.ylabel('Количество случаев', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('duration_distribution_simple.png', dpi=120)
plt.close()

# 3. Когда чаще всего работают? (по часам)
plt.figure(figsize=(10, 6))
df['hour'].value_counts().sort_index().plot(
    kind='line',
    marker='o',
    color='orange',
    linewidth=2
)
plt.title('В какое время чаще всего работают?', fontsize=14)
plt.xlabel('Час суток', fontsize=12)
plt.ylabel('Количество окон', fontsize=12)
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('work_hours_simple.png', dpi=120)
plt.close()

# 4. Соотношение количества окон и их продолжительности
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x='windows_count',
    y='duration_minutes',
    showfliers=False,
    palette='pastel'
)
plt.title('Как количество окон влияет на продолжительность?', fontsize=14)
plt.xlabel('Количество объединенных окон', fontsize=12)
plt.ylabel('Продолжительность (минуты)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('windows_vs_duration_simple.png', dpi=120)
plt.close()

# 5. Пример графика работ для одного перегона
if not df.empty:
    sample = df.iloc[0]['pereg_ms_id']
    sample_data = df[df['pereg_ms_id'] == sample].sort_values('window_start')

    plt.figure(figsize=(12, 4))
    for i, row in sample_data.iterrows():
        plt.plot(
            [row['window_start'], row['window_end']],
            [1, 1],
            linewidth=row['windows_count'] * 2,  # Чем больше окон - тем толще линия
            solid_capstyle='round',
            label=f"{row['window_start'].strftime('%H:%M')}-{row['window_end'].strftime('%H:%M')}"
        )

    plt.title(f'График работ для перегона {sample}', fontsize=14)
    plt.xlabel('Время', fontsize=12)
    plt.yticks([])
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), title='Время работ')
    plt.tight_layout()
    plt.savefig(f'peregon_schedule_{sample}.png', dpi=120, bbox_inches='tight')
    plt.close()

print("Готово! Созданы простые и понятные графики:")
print("1. top_peregons_simple.png - где больше всего окон")
print("2. duration_distribution_simple.png - как долго длятся работы")
print("3. work_hours_simple.png - в какое время чаще работают")
print("4. windows_vs_duration_simple.png - связь количества и продолжительности")
print("5. peregon_schedule_[ID].png - пример расписания работ")