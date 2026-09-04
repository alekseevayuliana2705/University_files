import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import sqrtm
import scipy
from scipy.integrate import trapezoid
import tkinter as tk
from tkinter import simpledialog, messagebox
from matplotlib.widgets import CheckButtons

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

def plot_ellipse(q, Q, color='pink', n=1000, alpha=1.0, linewidth=1.0):
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
        Цвет линии (детские цвета: pink, purple, lightblue)
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

def plot_ellipse3d(q, Q, color='pink', n=1000):
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
        Цвет линии (детские цвета: pink, purple, lightblue)
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

def plot_approx(appr, axis=np.array([0, 1]), color='pink', linewidth=2):
    """
    Визуализирует аппроксимацию множества достижимости.
    
    Параметры:
    ----------
    appr : ndarray
        Массив точек границы (2×n)
    axis : ndarray, optional
        Индексы координатных осей для подписей
    color : str, optional
        Цвет линии (детские цвета: pink, purple, lightblue)
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

# ============================================
# 3D ТРУБКА С ГАЛОЧКАМИ
# ============================================

def plot_tube_with_checkboxes(param, t_range, n_appr=50, axis=np.array([0, 1])):
    """
    Визуализирует трубку достижимости в 3D пространстве с галочками.
    
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
    """
    fig = plt.figure(figsize=(14, 10))
    fig.dpi = 100
    ax = fig.add_subplot(111, projection='3d')
    
    # Область для чекбоксов
    plt.subplots_adjust(bottom=0.15)
    ax_check = plt.axes([0.1, 0.02, 0.2, 0.1])
    
    # Галочки для выбора отображаемых оценок
    check = CheckButtons(ax_check, ['Внешние оценки', 'Внутренние оценки'], [True, True])
    
    # Создаем списки для хранения линий
    ext_lines = []
    int_lines = []
    
    # Сначала рисуем все линии (и внешние, и внутренние)
    for i in np.arange(0, t_range.size):
        t = t_range[i]
        x = t * np.ones(n_appr)
        
        # Внешние оценки - РОЗОВЫЙ
        appr_ext = get_approx(param, t, n_appr, 'ext', axis)
        line_ext, = ax.plot(x, appr_ext[0], appr_ext[1], 
                          color='hotpink', alpha=0.7, lw=1.0, 
                          visible=True, label='Внешние оценки' if i == 0 else "")
        ext_lines.append(line_ext)
        
        # Внутренние оценки - ФИОЛЕТОВЫЙ
        appr_int = get_approx(param, t, n_appr, 'int', axis)
        line_int, = ax.plot(x, appr_int[0], appr_int[1], 
                          color='mediumpurple', alpha=0.7, lw=1.0,
                          visible=True, label='Внутренние оценки' if i == 0 else "")
        int_lines.append(line_int)
    
    ax.set_xlabel('Время t')
    ax.set_ylabel(f'x{axis[0]+1}')
    ax.set_zlabel(f'x{axis[1]+1}')
    ax.set_title(f'Трубка достижимости ({len(t_range)} ободков)', fontsize=14)
    ax.legend()
    
    # Функция для обновления видимости
    def update_visibility(label):
        if label == 'Внешние оценки':
            visible = not ext_lines[0].get_visible()
            for line in ext_lines:
                line.set_visible(visible)
        elif label == 'Внутренние оценки':
            visible = not int_lines[0].get_visible()
            for line in int_lines:
                line.set_visible(visible)
        
        # Обновляем легенду
        handles, labels = ax.get_legend_handles_labels()
        visible_handles = [h for h in handles if h.get_visible()]
        visible_labels = [l for l, h in zip(labels, handles) if h.get_visible()]
        ax.legend(visible_handles, visible_labels)
        
        plt.draw()
    
    # Подключаем обработчик
    check.on_clicked(update_visibility)
    
    plt.show()

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

def plot_3d_proj(proj, axis=np.array([0, 1, 2]), color='lightblue'):
    """
    Визуализирует 3D поверхность эллипсоидальной оценки.
    
    Параметры:
    ----------
    proj : tuple
        Кортеж с координатами поверхности (x, y, z)
    axis : ndarray, optional
        Индексы координатных осей для подписей
    color : str, optional
        Цвет поверхности (детские цвета: pink, purple, lightblue)
    """
    fig = plt.figure()
    fig.dpi = 100
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(proj[0], proj[1], proj[2], alpha=0.75)    
    ax.set_xlabel('x' + str(axis[0] + 1))
    ax.set_ylabel('x' + str(axis[1] + 1))
    ax.set_zlabel('x' + str(axis[2] + 1))

# ============================================
# ВЫБОР РЕЖИМА 2D/3D
# ============================================

def choose_mode():
    """Выбор между 2D и 3D визуализацией"""
    root = tk.Tk()
    root.withdraw()
    
    choice = messagebox.askquestion("Выбор режима",
                                   "Какую визуализацию построить?\n\n"
                                   "Да - 3D трубка достижимости\n"
                                   "Нет - 2D графики")
    
    # Пример 1: 3D система
    A = lambda t: np.eye(3) * (t + 1)
    B = lambda t: np.array([[1, 2, 3],
                            [2, 0, -t],
                            [2, 2, 1]])
    q = lambda t: np.zeros(3)
    Q = lambda t: np.eye(3) * np.exp(t)
    dim = 3
    t0 = 0
    x0 = np.zeros(3)
    X0 = np.eye(3)
    grid_size = 200
    
    param = {'A': A, 'B': B, 'q': q, 'Q': Q, 't0': t0, 'x0': x0, 'X0': X0, 'gr_sz': grid_size, 'dim': dim}
    
    if choice == 'yes':
        # 3D ТРУБКА С ГАЛОЧКАМИ
        print("Запуск 3D трубки достижимости...")
        
        # Запрашиваем параметры трубки
        t_max = simpledialog.askfloat("Максимальное время", 
                                     "Введите максимальное время для трубки:",
                                     minvalue=0.5, maxvalue=5.0, initialvalue=2.0)
        if t_max is None:
            t_max = 2.0
        
        n_slices = simpledialog.askinteger("Количество ободков",
                                          "Введите количество временных срезов:",
                                          minvalue=10, maxvalue=100, initialvalue=30)
        if n_slices is None:
            n_slices = 30
        
        t_range = np.linspace(0.1, t_max, n_slices)
        
        # Выбор осей для проекции
        axis_choice = simpledialog.askinteger("Выбор осей",
                                             "Выберите оси для трубки:\n\n"
                                             "1 - оси x1 и x2\n"
                                             "2 - оси x1 и x3\n"
                                             "3 - оси x2 и x3",
                                             minvalue=1, maxvalue=3, initialvalue=1)
        
        if axis_choice == 1:
            axis = np.array([0, 1])
        elif axis_choice == 2:
            axis = np.array([0, 2])
        else:
            axis = np.array([1, 2])
        
        # Запускаем 3D трубку с галочками
        plot_tube_with_checkboxes(param, t_range, n_appr=50, axis=axis)
        
    else:
        # 2D ГРАФИКИ (оригинальные)
        print("Запуск 2D графиков...")
        
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
        grid_size = 2000
        
        param = {'A': A, 'B': B, 'q': q, 'Q': Q, 't0': t0, 'x0': x0, 'X0': X0, 'gr_sz': grid_size, 'dim': dim}
        
        t = 1
        appr = get_approx(param, t, n=200)
        
        # Внешние оценки - РОЗОВЫЙ
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.dpi = 100
        l_directions = 12
        phi = np.linspace(0, 2*np.pi, l_directions, endpoint=False)
        l = np.array([np.cos(phi), np.sin(phi)]).T
        
        for i in range(l_directions):
            center, config = get_ext_ellipse(param, t, l[i])
            plot_ellipse(center, config, 'pink', n=200, alpha=0.7, linewidth=1.5)
        
        ax.plot(appr[0], appr[1], c='hotpink', linewidth=3, label='Полная внешняя оценка')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_title('Внешние эллипсоидальные оценки')
        ax.legend()
        plt.show()
        
        # Внутренние оценки - ФИОЛЕТОВЫЙ
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.dpi = 100
        
        for i in range(l_directions):
            center, config = get_int_ellipse(param, t, l[i])
            plot_ellipse(center, config, 'purple', n=200, alpha=0.7, linewidth=1.5)
        
        ax.plot(appr[0], appr[1], c='mediumpurple', linewidth=3, label='Полная внутренняя оценка')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_title('Внутренние эллипсоидальные оценки')
        ax.legend()
        plt.show()
        
        # Сравнение внешних и внутренних оценок
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.dpi = 100
        
        # Внешние оценки - светлый розовый
        for i in range(l_directions):
            center, config = get_ext_ellipse(param, t, l[i])
            plot_ellipse(center, config, 'lightpink', n=200, alpha=0.4, linewidth=1.0)
        
        # Внутренние оценки - светлый фиолетовый
        for i in range(l_directions):
            center, config = get_int_ellipse(param, t, l[i])
            plot_ellipse(center, config, 'lavender', n=200, alpha=0.4, linewidth=1.0)
        
        # Полные оценки
        int_appr, ext_appr = get_approx(param, t, n=200, mode='both')
        ax.plot(ext_appr[0], ext_appr[1], c='deeppink', linewidth=3, label='Внешняя оценка')
        ax.plot(int_appr[0], int_appr[1], c='darkviolet', linewidth=3, label='Внутренняя оценка')
        
        ax.set_xlabel('x1', fontsize=12)
        ax.set_ylabel('x2', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(fontsize=12)
        ax.set_title('Сравнение внешних и внутренних оценок', fontsize=14)
        plt.show()

# ============================================
# ЗАПУСК ПРОГРАММЫ
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("ПРОГРАММА ДЛЯ ЭЛЛИПСОИДАЛЬНЫХ ОЦЕНОК МНОЖЕСТВА ДОСТИЖИМОСТИ")
    print("=" * 60)
    
    # Сразу спрашиваем 2D или 3D
    choose_mode()
    
    print("=" * 60)
    print("Программа завершена!")
    print("=" * 60)