import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

df = pd.read_csv('/Users/ulianaalekseeva/Desktop/my_prob.csv', low_memory=False)

binary_cols = ['need_for_2ab', 'relieve_voltage', 'with_wnd', 'scb_required',
               'approved', 'not_approved', 'rejected', 'voltage_sequence',
               'voltage_early', 'voltage_late']

for col in binary_cols:
    if col in df.columns:
        # Заменяем NaN на 0 и преобразуем в int
        df[col] = df[col].fillna(0).astype(int)

# Анализ корреляций только для числовых столбцов
numeric_df = df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(16, 12))
color = ['Red', "#FFFAAD", 'Green']
cmap = LinearSegmentedColormap.from_list("custom_gradient", color)
sns.heatmap(corr_matrix,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.5)
plt.title('Матрица корреляций между переменными', pad=20, fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('/Users/ulianaalekseeva/Desktop/correlation_matrix.png', dpi=300)
plt.close()

print("Анализ завершен. Результаты сохранены:")
print("correlation_matrix.png")