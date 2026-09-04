import pandas as pd
import numpy as np
import os

def is_positive_determinants(matrix):
    n = matrix.shape[0]
    for i in range(1, n+1):
        # Выделяем левый верхний угол размера i x i
        submatrix = matrix[:i, :i]
        # Вычисляем определитель этого подблока
        det = np.linalg.det(submatrix)
        # Если хотя бы один определитель <= 0, условие не выполняется
        if det <= 0:
            return False
    return True

# Путь к файлу в папке загрузок на Mac
downloads_path = os.path.expanduser("~/Downloads")
file_path = os.path.join(downloads_path, "Мексика 2016-2018 - Лист1-2.csv")

# Загрузка данных
df = pd.read_csv(file_path, encoding='utf-8')

# Фильтрация данных за 2016 год
df_2016 = df[df['YEAR'] == 2016]

# Получаем список всех отраслевых кодов (от A01_02 до T)
sectors = []
for col in df.columns:
    if col not in ['YEAR', 'NEW_COL', 'HFCE', 'NPISH', 'GGFC', 'GFCF', 'INVNT', 'DPABR', 'CONS_NONRES', 'EXPO', 'IMPO', 'TOTAL']:
        sectors.append(col)
        
# Отфильтруем только DOM_ строки (внутреннее производство)
dom_rows = df_2016[df_2016.iloc[:, 1].str.startswith('DOM_')]

# Преобразуем данные в числовой формат
dom_rows_numeric = dom_rows[sectors].apply(pd.to_numeric, errors='coerce')
dom_rows_numeric = dom_rows_numeric.fillna(0)  # Заменяем NaN на 0

# Матрица межотраслевых потоков X
X = dom_rows_numeric.values

# Вектор валового выпуска Y (из строки OUTPUT)
output_row = df_2016[df_2016.iloc[:, 1] == 'OUTPUT']
Y = output_row[sectors].apply(pd.to_numeric, errors='coerce').values.flatten()
Y = np.nan_to_num(Y)  # Заменяем NaN на 0

# Проверка размерностей
print(f"Размер матрицы X: {X.shape}")
print(f"Размер вектора Y: {Y.shape}")

# Проверяем, что нет нулей в знаменателе
if np.any(Y == 0):
    print("Внимание: есть нулевые значения в валовом выпуске!")
    # Добавляем маленькое значение чтобы избежать деления на ноль
    Y[Y == 0] = 1e-10

# Диагональная матрица с обратными значениями валового выпуска
Y_diag_inv = np.diag(1 / Y)

# Матрица коэффициентов прямых затрат A = X * Y^(-1)
A = X @ Y_diag_inv

# Создаем DataFrame с результатами
A_df = pd.DataFrame(A,
                   index=dom_rows.iloc[:, 1].str.replace('DOM_', ''),  # Названия отраслей
                   columns=sectors)                                    # Названия столбцов

# Сохраняем результат
output_file = os.path.join(downloads_path, "Матрица_Леонтьева_Мексика_2016.csv")
A_df.to_csv(output_file, encoding='utf-8')
print("Матрица Леонтьева успешно построена и сохранена!")
print(f"Файл сохранен: {output_file}")


I = np.eye(A.shape[0])
I_minus_A = I - A
try:
    I_minus_A_inv = np.linalg.inv(I_minus_A)
    print("Матрица (I - A) обратима.")

    # Проверка неотрицательности обратной матрицы
    if is_positive_determinants(I_minus_A):
        print("Все главные миноры (I - A) положительны. Матрица А продуктивная.")
    else:
        print("Матрица не продуктивная.")

except np.linalg.LinAlgError:
    print("Матрица (I - A) вырождена и необратима.")