import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

data = pd.read_csv('/Users/ulianaalekseeva/Desktop/my_prob.csv')

binary_cols_list = ['need_for_2ab', 'relieve_voltage', 'with_wnd', 'scb_required',
                   'approved', 'not_approved', 'rejected', 'voltage_sequence',
                   'voltage_early', 'voltage_late']

for column in binary_cols_list:
    if column in data.columns:
        data[column] = data[column].fillna(0)  # Заменяем пропуски на нули
        try:
            data[column] = data[column].astype(int)  # Пробуем сделать целыми числами
        except:
            print(f"Не получилось сделать {column} целыми числами.")

numbers_only = data.select_dtypes(include=['int', 'float'])

correlation_table = numbers_only.corr()

plt.figure(figsize=(15, 10))

color = ['Red', "#FFFAAD", 'Green']
cmap = LinearSegmentedColormap.from_list("custom_gradient", color)
heatmap = sns.heatmap(
    correlation_table,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    vmin=-1,
    vmax=1,
    linewidths=0.5
)
plt.title('Матрица корреляций между переменными', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()

heatmap.get_figure().savefig('/Users/ulianaalekseeva/Desktop/correlation_matrix.png', dpi=300)
plt.close()

print("Всё готово! Посмотрите файл на рабочем столе:")