import pandas as pd
import numpy as np
import os


def power_method(A, y0, eps=1e-6, max_iter=100):
    y = y0.copy()
    lambda_prev = 0
    for i in range(max_iter):
        # Умножаем матрицу на вектор
        y_new = np.dot(A, y)
        # Вычисляем новое собственное значение
        lambda_new = np.dot(y_new, y) / np.dot(y, y)
        # Проверяем сходимость
        if abs(lambda_new - lambda_prev) < eps:
            break
        y = y_new
        lambda_prev = lambda_new
    return lambda_new, y

base_path = "~/Downloads/"
files_2008 = {
    'original': "Матрица_Леонтьева_Мексика_2008.csv",
    'reduced': "Матрица_Леонтьева_Мексика_2008_редуцированная.csv",
    'aggregated': "Агрегированная_матрица_Леонтьева_Мексика_2008.csv"
}
files_2010 = {
    'original': "Матрица_Леонтьева_Мексика_2010.csv",
    'reduced': "Матрица_Леонтьева_Мексика_2010_редуцированная.csv",
    'aggregated': "Агрегированная_матрица_Леонтьева_Мексика_2010.csv"
}

results = {}

print("ВЫЧИСЛЕНИЕ СОБСТВЕННЫХ ЗНАЧЕНИЙ")

# 2008 год
print("\n2008 год:")
for matrix_type, filename in files_2008.items():
    file_path = os.path.expanduser(os.path.join(base_path, filename))
    df = pd.read_csv(file_path, index_col=0)
    matrix = df.values
    y0 = np.ones(matrix.shape[0])

    # Вычисляем собственное значение и вектор
    value, vector = power_method(matrix, y0)
    results[f"{matrix_type}_2008"] = value
    print(f"  {matrix_type}: λ = {value:.6f}")

# 2010 год
print("\n2010 год:")
for matrix_type, filename in files_2010.items():
    file_path = os.path.expanduser(os.path.join(base_path, filename))
    df = pd.read_csv(file_path, index_col=0)
    matrix = df.values
    y0 = np.ones(matrix.shape[0])
    value, vector = power_method(matrix, y0)
    results[f"{matrix_type}_2010"] = value

    print(f"  {matrix_type}: λ = {value:.6f}")

print()
print("ПРОВЕРКА СТАБИЛЬНОСТИ")

# Параметр для проверки стабильности
epsilon = 0.1

# Проверяем стабильность для 2008 года
print("\n2008 год:")
print("-" * 30)

# Сравнение оригинальной матрицы и матрицы исключения
if "original_2008" in results and "reduced_2008" in results:
    lambda1 = results["original_2008"]
    lambda2 = results["reduced_2008"]
    print("Сравнение оригинальной матрицы и матрицы исключения:")
    print(f"λ1 = {lambda1:.6f}")
    print(f"λ2 = {lambda2:.6f}")
    print(f"1/λ1 = {1 / lambda1:.6f}")
    print(f"1/λ2 = {1 / lambda2:.6f}")
    print(f"|1/λ1 - 1/λ2| = {abs(1 / lambda1 - 1 / lambda2):.6f}")
    if abs(1 / lambda1 - 1 / lambda2) < epsilon:
        print("Стабильность сохраняется")
    else:
        print("Нет стабильности")
    print()

# Сравнение оригинальной матрицы и агрегированной матрицы
if "original_2008" in results and "aggregated_2008" in results:
    lambda1 = results["original_2008"]
    lambda2 = results["aggregated_2008"]
    print("Сравнение оригинальной матрицы и агрегированной матрицы:")
    print(f"λ1 = {lambda1:.6f}")
    print(f"λ2 = {lambda2:.6f}")
    print(f"1/λ1 = {1 / lambda1:.6f}")
    print(f"1/λ2 = {1 / lambda2:.6f}")
    print(f"|1/λ1 - 1/λ2| = {abs(1 / lambda1 - 1 / lambda2):.6f}")
    if abs(1 / lambda1 - 1 / lambda2) < epsilon:
        print("Стабильность сохраняется")
    else:
        print("Нет стабильности")
    print()

# Проверяем стабильность для 2010 года
print("\n2010 год:")
print("-" * 30)

# Сравнение оригинальной матрицы и матрицы исключения
if "original_2010" in results and "reduced_2010" in results:
    lambda1 = results["original_2010"]
    lambda2 = results["reduced_2010"]
    print("Сравнение оригинальной матрицы и матрицы исключения:")
    print(f"λ1 = {lambda1:.6f}")
    print(f"λ2 = {lambda2:.6f}")
    print(f"1/λ1 = {1 / lambda1:.6f}")
    print(f"1/λ2 = {1 / lambda2:.6f}")
    print(f"|1/λ1 - 1/λ2| = {abs(1 / lambda1 - 1 / lambda2):.6f}")
    if abs(1 / lambda1 - 1 / lambda2) < epsilon:
        print("Стабильность сохраняется")
    else:
        print("Нет стабильности")
    print()

# Сравнение оригинальной матрицы и агрегированной матрицы
if "original_2010" in results and "aggregated_2010" in results:
    lambda1 = results["original_2010"]
    lambda2 = results["aggregated_2010"]
    print("Сравнение оригинальной матрицы и агрегированной матрицы:")
    print(f"λ1 = {lambda1:.6f}")
    print(f"λ2 = {lambda2:.6f}")
    print(f"1/λ1 = {1 / lambda1:.6f}")
    print(f"1/λ2 = {1 / lambda2:.6f}")
    print(f"|1/λ1 - 1/λ2| = {abs(1 / lambda1 - 1 / lambda2):.6f}")
    if abs(1 / lambda1 - 1 / lambda2) < epsilon:
        print("Стабильность сохраняется")
    else:
        print("Нет стабильности")
    print()