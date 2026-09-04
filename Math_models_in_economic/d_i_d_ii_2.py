import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


# Загрузка матрицы
A = pd.read_csv("~/Downloads/Матрица_Леонтьева_Мексика_2010.csv", index_col=0)

# Описания отраслей
descriptions = {
    'A01_02': 'Agriculture, hunting, forestry',
    'A03': 'Fishing and aquaculture',
    'B05_06': 'Mining and quarrying, energy producing products',
    'B07_08': 'Mining and quarrying, non-energy producing products',
    'B09': 'Mining support service activities',
    'C10T12': 'Food products, beverages and tobacco',
    'C13T15': 'Textiles, textile products, leather and footwear',
    'C16': 'Wood and products of wood and cork',
    'C17_18': 'Paper products and printing',
    'C19': 'Coke and refined petroleum products',
    'C20': 'Chemical and chemical products',
    'C21': 'Pharmaceuticals, medicinal chemical and botanical products',
    'C22': 'Rubber and plastics products',
    'C23': 'Other non-metallic mineral products',
    'C24': 'Basic metals',
    'C25': 'Fabricated metal products',
    'C26': 'Computer, electronic and optical equipment',
    'C27': 'Electrical equipment',
    'C28': 'Machinery and equipment, nec',
    'C29': 'Motor vehicles, trailers and semi-trailers',
    'C30': 'Other transport equipment',
    'C31T33': 'Manufacturing nec; repair and installation of machinery and equipment',
    'D': 'Electricity, gas, steam and air conditioning supply',
    'E': 'Water supply; sewerage, waste management and remediation activities',
    'F': 'Construction',
    'G': 'Wholesale and retail trade; repair of motor vehicles',
    'H49': 'Land transport and transport via pipelines',
    'H50': 'Water transport',
    'H51': 'Air transport',
    'H52': 'Warehousing and support activities for transportation',
    'H53': 'Postal and courier activities',
    'I': 'Accommodation and food service activities',
    'J58T60': 'Publishing, audiovisual and broadcasting activities',
    'J61': 'Telecommunications',
    'J62_63': 'IT and other information services',
    'K': 'Financial and insurance activities',
    'L': 'Real estate activities',
    'M': 'Professional, scientific and technical activities',
    'N': 'Administrative and support services',
    'O': 'Public administration and defence; compulsory social security',
    'P': 'Education',
    'Q': 'Human health and social work activities',
    'R': 'Arts, entertainment and recreation',
    'S': 'Other service activities',
    'T': 'Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use'
}

# Функция для вычисления d_i и d_ii
def create_d(A):
    industries = A.index
    n = len(industries)

    # Инициализируем списки для результатов
    d_i_values = []
    d_ii_values = []

    # Вычисляем d_i для каждой отрасли i
    for i in range(n):
        sum_d_i = 0.0
        # Суммируем a_{i,j} для всех j != i
        for j in range(n):
            if j != i:  # исключаем диагональные элементы
                sum_d_i += A.iloc[i, j]
        d_i_values.append(sum_d_i)

    # Вычисляем d_ii для каждой отрасли i
    for i in range(n):
        sum_d_ii = 0.0
        # Первая сумма: по j (j != i)
        for j in range(n):
            if j != i:
                # Вторая сумма: по k
                for k in range(n):
                    sum_d_ii += A.iloc[i, k] * A.iloc[k, j]
        d_ii_values.append(sum_d_ii)

    # Создаем Series с результатами
    d_i = pd.Series(d_i_values, index=industries, name='d_i')
    d_ii = pd.Series(d_ii_values, index=industries, name='d_ii')
    return d_i, d_ii

# Вычисляем характеристики
d_i, d_ii = create_d(A)
d_i_clean = {k: float(v) for k, v in d_i.items()}
d_ii_clean = {k: float(v) for k, v in d_ii.items()}

# Находим топовые отрасли
print("ТОП-4 отрасли по величине d_i:")
top4_d_i = sorted(d_i_clean.items(), key=lambda x: x[1], reverse=True)[:4]
for i, (industry, value) in enumerate(top4_d_i, 1):
    print(f"{i}. {industry} ({descriptions.get(industry)}): {value:.4f}")

print("\nТОП-2 отрасли по величине d_ii:")
top2_d_ii = sorted(d_ii_clean.items(), key=lambda x: x[1], reverse=True)[:2]
for i, (industry, value) in enumerate(top2_d_ii, 1):
    print(f"{i}. {industry} ({descriptions.get(industry)}): {value:.4f}")


# Упрощенная функция графа
def create_simple_graph(d, A, title='Граф', epsilon=0.03):
    indices = list(A.index)
    n = len(indices)

    # Создаем граф
    G = nx.DiGraph()

    # Добавляем вершины с размерами
    for i, industry in enumerate(indices):
        G.add_node(industry, size=300 + d[industry] * 2000)

    # Добавляем ребра только для связей >= epsilon
    for i in range(n):
        for j in range(n):
            if i != j and A.iloc[i, j] >= epsilon:
                G.add_edge(indices[i], indices[j])

    # Рисуем граф
    plt.figure(figsize=(12, 8))
    pos = nx.circular_layout(G)

    node_sizes = [G.nodes[node]['size'] for node in G.nodes()]

    nx.draw(G, pos,
            node_size=node_sizes,
            node_color='skyblue',
            with_labels=True,
            font_size=8,
            arrows=True,
            arrowsize=15,
            edge_color='lightblue')

    plt.title(title)
    plt.show()


create_simple_graph(d_i_clean, A, 'Граф для d_i (1й порядок)', epsilon=0.01)
create_simple_graph(d_ii_clean, A, 'Граф для d_ii (2й порядок)', epsilon=0.01)