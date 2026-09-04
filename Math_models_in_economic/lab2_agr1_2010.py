import pandas as pd
import numpy as np
import os

# Загружаем вектор валового выпуска
downloads_path = os.path.expanduser("~/Downloads")
file_path = os.path.join(downloads_path, "Мексика 2008-2010 - Лист1.csv")
df_output = pd.read_csv(file_path)

# Получаем список всех отраслевых кодов (от A01_02 до T)
sectors = []
for col in df_output.columns:
    if col not in ['YEAR', 'NEW_COL', 'HFCE', 'NPISH', 'GGFC', 'GFCF', 'INVNT', 'DPABR', 'CONS_NONRES', 'EXPO', 'IMPO',
                   'TOTAL']:
        sectors.append(col)

# Вектор валового выпуска для 2010 года
output_row = df_output[df_output.iloc[:, 1] == 'OUTPUT']
x = output_row[sectors].apply(pd.to_numeric, errors='coerce').values.flatten()
x = np.nan_to_num(x)

file_path = "~/Downloads/Матрица_Леонтьева_Мексика_2010.csv"
df = pd.read_csv(file_path, index_col=0)

codes = df.index.tolist()
A = df.values
common_codes = list(set(codes) & set(sectors))

# Создаем новые массивы только с общими отраслями
common_indices_codes = [codes.index(code) for code in common_codes]
common_indices_sectors = [sectors.index(code) for code in common_codes]
A_common = A[common_indices_codes, :][:, common_indices_codes]
x_common = x[common_indices_sectors]
codes_common = [codes[i] for i in common_indices_codes]

# Группы отраслей
group1 = [code for code in ['A01_02', 'A03', 'B05_06', 'B07_08', 'B09', 'C10T12', 'C13T15', 'E'] if code in common_codes]
group2 = [code for code in ['C16', 'C17_18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25',
          'C26', 'C27', 'C28', 'C29', 'C30', 'C31T33', 'F', 'D'] if code in common_codes]
group3 = [code for code in ['G', 'H49', 'H50', 'H51', 'H52', 'H53', 'I', 'R'] if code in common_codes]
group4 = [code for code in ['J58T60', 'J61', 'J62_63', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'S', 'T'] if code in common_codes]
all_groups = [group1, group2, group3, group4]

# Функция для нахождения номеров строк
def find_row_numbers(codes, group_codes):
    row_numbers = []
    for code in group_codes:
        if code in codes:
            index = codes.index(code)
            row_numbers.append(index)
    return row_numbers

# Получаем индексы для всех групп
group1_rows = find_row_numbers(codes_common, group1)
group2_rows = find_row_numbers(codes_common, group2)
group3_rows = find_row_numbers(codes_common, group3)
group4_rows = find_row_numbers(codes_common, group4)
all_group_rows = [group1_rows, group2_rows, group3_rows, group4_rows]

# Агрегирование по теоретической формуле
new_matrix = np.zeros((4, 4))
for k in range(4):
    for l in range(4):
        numerator = 0
        denominator = 0
        rows_k = all_group_rows[k]  # I_k
        cols_l = all_group_rows[l]  # I_l

        # Вычисляем числитель
        for i in rows_k:
            for j in cols_l:
                numerator += A_common[i, j] * x_common[j]

        # Вычисляем знаменатель
        for j in cols_l:
            denominator += x_common[j]

        # Вычисляем элемент агрегированной матрицы
        if denominator != 0:
            new_matrix[k, l] = numerator / denominator
        else:
            new_matrix[k, l] = 0

print("Новая агрегированная матрица размером 4x4:")
print(np.round(new_matrix, 6))
print()

group_names = [
    "Сельское хозяйство и пищевая промышленность",
    "Промышленность и строительство",
    "Транспорт, туризм и торговля",
    "Услуги, финансы и государственный сектор"
]
print("Состав новых групп:")
for i, (name, rows, codes_list) in enumerate(zip(group_names, all_group_rows, all_groups)):
    print(f"Группа {i + 1}: {name}")
    print(f"Входит отраслей: {len(rows)}")
    print(f"Коды отраслей: {codes_list}")
    print()
result_df = pd.DataFrame(new_matrix, index=group_names, columns=group_names)
output_file = "~/Downloads/Агрегированная_матрица_Леонтьева_Мексика_2010.csv"
result_df.to_csv(output_file)
print(f"Агрегированная матрица сохранена в файл: {output_file}")