import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import sqrtm
import scipy
from scipy.integrate import trapezoid

def fund_sol(A, tau_range):
    """
    Вычисляет фундаментальную матрицу решения для линейной системы.
    
    Фундаментальная матрица X(t, τ) удовлетворяет уравнению:
        dX/dt = X * A(t), X(τ, τ) = I
    
    Параметры:
    ----------
    A : callable
        Функция, возвращающая матрицу системы A(t) размера n×n
    tau_range : array_like
        Временной диапазон [τ0, τ1, ..., τk] для вычисления решения
        
    Возвращает:
    -----------
    ndarray
        3D-массив размера (len(tau_range), n, n), где 
        res[i] = X(τ_i, τ_0) - фундаментальная матрица от τ_i до τ_0
    """
    n = A(0).shape[0]
    ode_func = lambda y, t: y @ A(t)
    res = np.zeros((tau_range.size, n, n))
    for i in np.arange(0, n):
        y0 = np.eye(n)[i]
        res[:, i, :] = np.flip(scipy.integrate.odeint(ode_func, y0, -np.flip(tau_range)), axis = 0)
    return res

def get_ext_ellipse(param, t, l):
    """
    Вычисляет внешнюю эллипсоидальную оценку множества достижимости.
    
    Внешняя оценка всегда содержит истинное множество достижимости.
    Основана на методе опорных функций и дает оценку вида:
        E(q_ext, Q_ext) ⊇ Reach(t)
    
    Параметры:
    ----------
    param : dict
        Словарь параметров системы
    t : float
        Конечный момент времени
    l : ndarray
        Направляющий вектор (единичной длины)
        
    Возвращает:
    -----------
    tuple (center, config)
        center : ndarray - центр эллипсоида
        config : ndarray - конфигурационная матрица Q (положительно определенная)
    """
    sz = param['gr_sz']
    n = param['dim']
    t_range = np.linspace(param['t0'], t, sz)
    X_fund = fund_sol(param['A'], t_range)
    p0 = np.sqrt(np.dot(l, X_fund[0] @ param['X0'] @ X_fund[0].T @ l))
    p = np.zeros(sz) 
    int_f1 = np.zeros((sz, n, n))
    int_f2 = np.zeros((sz, n))
    B = param['B']
    Q = param['Q']
    q = param['q']
    for i in np.arange(0, sz):
        tau = t_range[i]
        p[i] = np.sqrt(np.dot(l, X_fund[i] @ B(tau) @ Q(tau) @ B(tau).T @ X_fund[i].T @ l))
        int_f1[i] = X_fund[i] @ B(tau) @ Q(tau) @ B(tau).T @ X_fund[i].T / p[i]
        int_f2[i] = X_fund[i] @ B(tau) @ q(tau)
    center = X_fund[0] @ param['x0'] + trapezoid(int_f2, t_range, axis=0)
    config = (p0 + trapezoid(p, t_range)) * (X_fund[0] @ param['X0'] @ X_fund[0].T / p0 + trapezoid(int_f1, t_range, axis=0))
    return (center, config)

def get_int_ellipse(param, t, l):
    """
    Вычисляет внутреннюю эллипсоидальную оценку множества достижимости.
    
    Внутренняя оценка всегда содержится в истинном множестве достижимости.
    Использует метод сопряжения эллипсоидов:
        E(q_int, Q_int) ⊆ Reach(t)
    
    Параметры:
    ----------
    param : dict
        Словарь параметров системы
    t : float
        Конечный момент времени
    l : ndarray
        Направляющий вектор (единичной длины)
        
    Возвращает:
    -----------
    tuple (center, config)
        center : ndarray - центр эллипсоида
        config : ndarray - конфигурационная матрица Q (положительно определенная)
    """
    sz = param['gr_sz']
    n = param['dim']
    t_range = np.linspace(param['t0'], t, sz)
    X_fund = fund_sol(param['A'], t_range)
    B = param['B']
    Q = param['Q']
    q = param['q']
    int_f = np.zeros((sz, n, n))
    int_fc = np.zeros((sz, n))
    for i in np.arange(0, sz):
        tau = t_range[i]
        int_fc[i] = X_fund[i] @ B(tau) @ q(tau)
        Q_sq = sqrtm(Q(tau))
        lam = np.linalg.norm(Q_sq @ B(tau).T @ X_fund[i].T @ l) / np.linalg.norm(sqrtm(param['X0']) @ X_fund[0].T @ l)
        a = Q_sq @ B(tau).T @ X_fund[i].T @ l
        b = lam * sqrtm(param['X0']) @ X_fund[0].T @ l
        Ua, da, Va = np.linalg.svd(np.reshape(a, (n, 1)))
        Ub, db, Vb = np.linalg.svd(np.reshape(b, (n, 1)))
        S = Vb / Va * Ub @ Ua.T
        int_f[i] = S @ Q_sq @ B(tau).T @ X_fund[i].T
        
    Q_ast = sqrtm(param['X0']) @ X_fund[0].T + trapezoid(int_f, t_range, axis=0)
    config = Q_ast.T @ Q_ast
    center = X_fund[0] @ param['x0'] + trapezoid(int_fc, t_range, axis=0)    
    return center, config

def plot_ellipse(q, Q, color='b', n=1000, alpha=1.0, linewidth=1.0):
    """
    Визуализирует 2D эллипсоид.
    
    Эллипсоид задается в виде:
        E = {x: (x-q)ᵀ Q⁻¹ (x-q) ≤ 1}
    
    Параметры:
    ----------
    q : ndarray
        Центр эллипсоида (размерности 2)
    Q : ndarray
        Конфигурационная матрица 2×2 (положительно определенная)
    color : str, optional
        Цвет линии
    n : int, optional
        Количество точек для аппроксимации эллипса
    alpha : float, optional
        Прозрачность линии
    linewidth : float, optional
        Толщина линии
    """
    phi = np.linspace(0, 2 * np.pi, n)
    l = np.array([np.cos(phi), np.sin(phi)])
    x = Q @ l / np.sqrt(np.sum(l * (Q @ l), axis=0))
    plt.plot(x[0] + q[0], x[1] + q[1], c=color, alpha=alpha, linewidth=linewidth)

def plot_ellipse3d(q, Q, color='b', n=1000):
    """
    Визуализирует проекцию 3D эллипсоида на координатные плоскости.
    
    Примечание: Функция рисует 2D проекцию 3D эллипсоида.
    
    Параметры:
    ----------
    q : ndarray
        Центр эллипсоида (размерности 3)
    Q : ndarray
        Конфигурационная матрица 3×3
    color : str, optional
        Цвет линии
    n : int, optional
        Количество точек для аппроксимации
    """
    phi = np.linspace(0, 2 * np.pi, n)
    l = np.array([np.cos(phi), np.sin(phi)])
    x = Q @ l / np.sqrt(np.sum(l * (Q @ l), axis=0))
    plt.plot(x[0] + q[0], x[1] + q[1], x[2] + q[2], c=color)

def support_func(q, Q, l):
    """
    Вычисляет опорную функцию эллипсоида в заданном направлении.
    
    Опорная функция эллипсоида E(q, Q) в направлении l:
        ρ(l|E) = max{x·l: x ∈ E} = q·l + √(lᵀ Q l)
    
    Параметры:
    ----------
    q : ndarray
        Центр эллипсоида
    Q : ndarray
        Конфигурационная матрица
    l : ndarray
        Направляющий вектор
        
    Возвращает:
    -----------
    ndarray
        Точка на границе эллипсоида, максимальная в направлении l
    """
    return q + Q @ l / np.sqrt(np.dot(l, Q @ l))

def get_approx(param, t, n=100, mode='ext', axis=np.array([0, 1])):
    """
    Вычисляет аппроксимацию полного множества достижимости.
    
    Метод основан на вычислении опорных точек в равномерно
    распределенных направлениях и построении их выпуклой оболочки.

    Берет n равномерно распределенных направлений в плоскости

    Для каждого направления вычисляет опорную точку (внешнюю или внутреннюю)

    Строит выпуклую оболочку этих точек
    
    Параметры:
    ----------
    param : dict
        Словарь параметров системы
    t : float
        Конечный момент времени
    n : int, optional
        Количество направлений для аппроксимации
    mode : str, optional
        Режим вычисления: 'int' - внутренняя, 'ext' - внешняя, 
        'both' - обе оценки
    axis : ndarray, optional
        Индексы координатных осей для проекции
        
    Возвращает:
    -----------
    ndarray или tuple
        - Если mode='int' или 'ext': массив точек границы (2×n)
        - Если mode='both': кортеж (int_appr, ext_appr)
    """
    phi = np.linspace(0, 2 * np.pi, n)
    int_appr = np.zeros((2, n))
    ext_appr = np.zeros((2, n))
    for i in np.arange(0, n):
        l = np.eye(param['dim'])[axis[0]] * np.cos(phi[i]) + np.eye(param['dim'])[axis[1]] * np.sin(phi[i])
        if mode == 'both' or mode == 'int':
            c_int, Q_int = get_int_ellipse(param, t, l)
            int_appr[:, i] = support_func(c_int, Q_int, l)[axis]
        if mode == 'both' or mode == 'ext':
            c_ext, Q_ext = get_ext_ellipse(param, t, l)
            ext_appr[:, i] = support_func(c_ext, Q_ext, l)[axis]
    if mode == 'int':
        return int_appr
    if mode == 'ext':
        return ext_appr
    return (int_appr, ext_appr)

def plot_approx(appr, axis=np.array([0, 1]), color='b', linewidth=2):
    """
    Визуализирует аппроксимацию множества достижимости.
    
    Параметры:
    ----------
    appr : ndarray
        Массив точек границы (2×n)
    axis : ndarray, optional
        Индексы координатных осей для подписей
    color : str, optional
        Цвет линии
    linewidth : float, optional
        Толщина линии
    """
    fig, ax = plt.subplots()
    fig.dpi = 70
    ax.plot(appr[0], appr[1], c=color, linewidth=linewidth)
    ax.set_xlabel('x' + str(axis[0] + 1))
    ax.set_ylabel('x' + str(axis[1] + 1))
    ax.grid()
    ax.axis('equal')

def plot_tube(param, t_range, n_appr, axis=np.array([0, 1]), color='b'):
    """
    Визуализирует трубку достижимости в 3D пространстве.
    
    Трубка представляет собой объединение множеств достижимости
    для разных моментов времени.
    
    Параметры:
    ----------
    param : dict
        Словарь параметров системы
    t_range : array_like
        Диапазон моментов времени
    n_appr : int
        Количество точек для аппроксимации каждого сечения
    axis : ndarray, optional
        Индексы координатных осей для проекции
    color : str, optional
        Цвет линии
    """
    fig = plt.figure(figsize=(12, 8))
    fig.dpi = 100
    ax = fig.add_subplot(111, projection='3d')
    for i in np.arange(0, t_range.size):
        x = t_range[i] * np.ones(n_appr)
        appr = get_approx(param, t_range[i], n_appr, 'ext', axis)
        y = appr[0]
        z = appr[1]
        ax.plot(x, y, z, color=color, alpha=0.85, lw=1.0)
        ax.set_xlabel('t')
        ax.set_ylabel('x' + str(axis[0] + 1))
        ax.set_zlabel('x' + str(axis[1] + 1))
    plt.title('Трубка достижимости (50 ободков)', fontsize=14)

def get_3d_proj(param, t, n, axis=np.array([0, 1, 2])):
    """
    Вычисляет 3D поверхность эллипсоидальной оценки.
    
    Использует сферические координаты для равномерного
    покрытия единичной сферы направлениями.
    
    Параметры:
    ----------
    param : dict
        Словарь параметров системы
    t : float
        Конечный момент времени
    n : int
        Разрешение по углам
    axis : ndarray, optional
        Индексы координатных осей
        
    Возвращает:
    -----------
    tuple
        Три массива (x, y, z) координат поверхности
    """
    phi = np.linspace(0, 2 * np.pi, 2 * n)
    psi = np.linspace(-np.pi / 2, np.pi / 2, n)
    x = np.zeros((2 * n, n))
    y = np.zeros((2 * n, n))
    z = np.zeros((2 * n, n))

    for (i, a1) in enumerate(phi):
        for (j, a2) in enumerate(psi):
            l = np.cos(a1) * np.cos(a2) * np.eye(param['dim'])[axis[0]] + np.sin(a1) * np.cos(a2) * np.eye(param['dim'])[axis[1]] + np.sin(a2) * np.eye(param['dim'])[axis[2]]
            c_ext, Q_ext = get_ext_ellipse(param, t, l)
            [x[i, j], y[i, j], z[i, j]] = support_func(c_ext, Q_ext, l)[axis]
    return (x, y, z)

def plot_3d_proj(proj, axis=np.array([0, 1, 2]), color='purple'):
    """
    Визуализирует 3D поверхность эллипсоидальной оценки.
    
    Параметры:
    ----------
    proj : tuple
        Кортеж с координатами поверхности (x, y, z)
    axis : ndarray, optional
        Индексы координатных осей для подписей
    color : str, optional
        Цвет поверхности
    """
    fig = plt.figure()
    fig.dpi = 100
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(proj[0], proj[1], proj[2], alpha=0.75)    
    ax.set_xlabel('x' + str(axis[0] + 1))
    ax.set_ylabel('x' + str(axis[1] + 1))
    ax.set_zlabel('x' + str(axis[2] + 1))

# Исправленные параметры с правильными размерами
A = lambda t: np.eye(3) * (t + 1)
B = lambda t: np.array([[1, 2, 3],    # 3x3 матрица
                        [2, 0, -t],
                        [2, 2, 1]])
q = lambda t: np.zeros(3)             # вектор длины 3
Q = lambda t: np.eye(3) * np.exp(t)   # 3x3 матрица
dim = 3
t0 = 0
x0 = np.zeros(3)
X0 = np.eye(3)
grid_size = 200  # Увеличено для гладкости

param = {'A': A, 'B': B, 'q': q, 'Q': Q, 't0': t0, 'x0': x0, 'X0': X0, 'gr_sz': grid_size, 'dim': dim}

# Тестирование с увеличенными параметрами для гладкости
try:
    print("Тестирование внешней оценки...")
    l_test = np.array([1, 0, 0])
    center, config = get_ext_ellipse(param, 0.5, l_test)
    print(f"Успешно: center={center}, config shape={config.shape}")
    
    print("Тестирование аппроксимации...")
    appr = get_approx(param, 0.5, n=100)  # Увеличено с 20 до 100
    plot_approx(appr, linewidth=2)
    plt.title("Тестовая аппроксимация (гладкая)")
    plt.show()
    
    print("Тестирование трубки с 50 ободками...")
    plot_tube(param, np.linspace(0.1, 1, 50), 80, np.array([0, 1]), 'g')  # 50 ободков
    plt.show()
    
except Exception as e:
    print(f"Ошибка: {e}")
    import traceback
    traceback.print_exc()

# Вторая система с увеличенными параметрами для гладкости
A = lambda t: np.array([[t, 1],
                        [0, -1.5]])
B = lambda t: np.array([[1, 0.3],
                        [0, 1 + np.exp(t)]]) 
q = lambda t: np.zeros(2)
Q = lambda t: np.array([[1, -0.5 * t**2],
                        [-0.5 * t**2, 1]])
dim = A(0).shape[0]
t0 = 0
x0 = np.zeros(2)
X0 = np.eye(2)
grid_size = 2000  # Значительно увеличено для гладкости

param = {'A': A, 'B': B, 'q': q, 'Q': Q, 't0': t0, 'x0': x0, 'X0': X0, 'gr_sz': grid_size, 'dim': dim}

t = 1
appr = get_approx(param, t, n=200)  # Увеличено для гладкости

# внешние оценки - больше направлений для гладкости
fig, ax = plt.subplots(figsize=(10, 8))
fig.dpi = 100
l_directions = 12  # Увеличено количество направлений
phi = np.linspace(0, 2*np.pi, l_directions, endpoint=False)
l = np.array([np.cos(phi), np.sin(phi)]).T

for i in range(l_directions):
    center, config = get_ext_ellipse(param, t, l[i])
    plot_ellipse(center, config, 'purple', n=200, alpha=0.7, linewidth=1.5)

ax.plot(appr[0], appr[1], c='blue', linewidth=3, label='Полная внешняя оценка')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.set_title('Внешние эллипсоидальные оценки (гладкие)')
ax.legend()
plt.show()

# внутренние оценки - больше направлений для гладкости
fig, ax = plt.subplots(figsize=(10, 8))
fig.dpi = 100

for i in range(l_directions):
    center, config = get_int_ellipse(param, t, l[i])
    plot_ellipse(center, config, 'red', n=200, alpha=0.7, linewidth=1.5)

ax.plot(appr[0], appr[1], c='blue', linewidth=3, label='Полная внутренняя оценка')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.set_title('Внутренние эллипсоидальные оценки (гладкие)')
ax.legend()
plt.show()

# Сравнение внешних и внутренних оценок на одном графике
fig, ax = plt.subplots(figsize=(12, 10))
fig.dpi = 100

# Внешние оценки
for i in range(l_directions):
    center, config = get_ext_ellipse(param, t, l[i])
    plot_ellipse(center, config, 'lightblue', n=200, alpha=0.4, linewidth=1.0)

# Внутренние оценки
for i in range(l_directions):
    center, config = get_int_ellipse(param, t, l[i])
    plot_ellipse(center, config, 'lightcoral', n=200, alpha=0.4, linewidth=1.0)

# Полные оценки
int_appr, ext_appr = get_approx(param, t, n=200, mode='both')
ax.plot(ext_appr[0], ext_appr[1], c='darkblue', linewidth=3, label='Внешняя оценка')
ax.plot(int_appr[0], int_appr[1], c='darkred', linewidth=3, label='Внутренняя оценка')

ax.set_xlabel('x1', fontsize=12)
ax.set_ylabel('x2', fontsize=12)
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.legend(fontsize=12)
ax.set_title('Сравнение внешних и внутренних оценок', fontsize=14)
plt.show()
