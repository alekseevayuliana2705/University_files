import pandas as pd
import numpy as np

# Загрузка данных
df = pd.read_csv("~/Downloads/Матрица_Леонтьева_Мексика_2010.csv", index_col=0)
print("Столбцы в файле:", df.columns.tolist())

# Получаем коды отраслей и матрицу A
codes = df.index.tolist()
A = df.to_numpy()  # Вся матрица

print(f"Размер матрицы: {A.shape}")
print(f"Количество отраслей: {len(codes)}")


# Функция удаления отраслей с наименьшими связями
def remove_matrix(A, codes, num_remove=10):
    links = {}
    for i in range(len(A)):
        # Сумма строки без диагонального элемента
        row_sum = A[i].sum() - A[i, i]
        links[i] = row_sum

    to_remove = sorted(links, key=links.get)[:num_remove]
    B = np.delete(A, to_remove, axis=0)
    B = np.delete(B, to_remove, axis=1)
    B_codes = [codes[i] for i in range(len(codes)) if i not in to_remove]

    print("\nУдалённые отрасли:")
    for i in to_remove:
        print(f"{codes[i]} ({links[i]:.6f})")

    return B, B_codes


# Применение функции
B, B_codes = remove_matrix(A, codes)

print(f"\nИсходный размер: {A.shape}")
print(f"Новый размер: {B.shape}")
print(f"Осталось отраслей: {len(B_codes)}")

# Сохранение результата (опционально)
result_df = pd.DataFrame(B, index=B_codes, columns=B_codes)
result_df.to_csv("~/Downloads/Матрица_Леонтьева_Мексика_2010_редуцированная.csv")
print("\nРедуцированная матрица сохранена в файл: Матрица_Леонтьева_Мексика_2010_редуцированная.csv")