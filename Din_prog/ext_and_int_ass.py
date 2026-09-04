import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from scipy.integrate import odeint
from scipy.linalg import sqrtm
import tkinter as tk
from tkinter import simpledialog, messagebox


class EllipsoidalEstimator3D:
    def __init__(self):
        self.param = None
        self.current_t = 1.0
        self.display_states = {
            'reachable': True,  # Множество достижимости (аппроксимация)
            'internal': True,
            'external': True
        }
        self.fig = None
        self.ax = None
        self.proj_int = None
        self.proj_ext = None
        self.proj_reachable = None

    def fund_sol(self, A, tau_range):
        """Фундаментальная матрица"""
        n = A(0).shape[0]
        ode_func = lambda y, t: y @ A(t)
        res = np.zeros((tau_range.size, n, n))
        for i in np.arange(0, n):
            y0 = np.eye(n)[i]
            res[:, i, :] = np.flip(odeint(ode_func, y0, -np.flip(tau_range)), axis=0)
        return res

    def get_ext_ellipse(self, param, t, l):
        """Внешняя эллипсоидальная оценка"""
        sz = param['gr_sz']
        n = param['dim']
        t_range = np.linspace(param['t0'], t, sz)
        X_fund = self.fund_sol(param['A'], t_range)

        # Вычисляем p0 и p(τ) по формулам
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

        # Центр внешней оценки
        center = X_fund[0] @ param['x0'] + np.trapz(int_f2, t_range, axis=0)

        # Матрица конфигурации внешней оценки
        config = (p0 + np.trapz(p, t_range)) * (
                X_fund[0] @ param['X0'] @ X_fund[0].T / p0 + np.trapz(int_f1, t_range, axis=0))

        return center, config

    def get_int_ellipse(self, param, t, l):
        """Внутренняя эллипсоидальная оценка"""
        sz = param['gr_sz']
        n = param['dim']
        t_range = np.linspace(param['t0'], t, sz)
        X_fund = self.fund_sol(param['A'], t_range)
        B = param['B']
        Q = param['Q']
        q = param['q']
        int_f = np.zeros((sz, n, n))
        int_fc = np.zeros((sz, n))

        for i in np.arange(0, sz):
            tau = t_range[i]
            int_fc[i] = X_fund[i] @ B(tau) @ q(tau)
            Q_sq = sqrtm(Q(tau))

            # Вычисляем λ(τ) по формуле
            numerator = np.linalg.norm(Q_sq @ B(tau).T @ X_fund[i].T @ l)
            denominator = np.linalg.norm(sqrtm(param['X0']) @ X_fund[0].T @ l)
            lam = numerator / denominator if denominator > 1e-10 else 0

            a = Q_sq @ B(tau).T @ X_fund[i].T @ l
            b = lam * sqrtm(param['X0']) @ X_fund[0].T @ l

            # Сингулярное разложение для нахождения S(τ)
            Ua, da, Va = np.linalg.svd(np.reshape(a, (n, 1)))
            Ub, db, Vb = np.linalg.svd(np.reshape(b, (n, 1)))

            # Вычисляем ортогональную матрицу S(τ)
            S = Vb[0, 0] / Va[0, 0] * Ub @ Ua.T
            int_f[i] = S @ Q_sq @ B(tau).T @ X_fund[i].T

        # Матрица Q*(t) для внутренней оценки
        Q_ast = sqrtm(param['X0']) @ X_fund[0].T + np.trapz(int_f, t_range, axis=0)
        config = Q_ast.T @ Q_ast
        center = X_fund[0] @ param['x0'] + np.trapz(int_fc, t_range, axis=0)
        return center, config

    def support_func(self, q, Q, l):
        """Опорная функция эллипсоида"""
        Ql = Q @ l
        lQl = np.dot(l, Ql)
        if lQl < 1e-10:
            return q
        return q + Ql / np.sqrt(lQl)

    def get_reachable_set_approximation(self, param, t, n=15, axis=np.array([0, 1, 2])):
        """Аппроксимация множества достижимости через объединение внутренних оценок"""
        phi = np.linspace(0, 2 * np.pi, 2 * n)
        psi = np.linspace(-np.pi / 2, np.pi / 2, n)
        x_reach = np.zeros((2 * n, n))
        y_reach = np.zeros((2 * n, n))
        z_reach = np.zeros((2 * n, n))

        # Используем объединение внутренних оценок по разным направлениям
        # как лучшую аппроксимацию множества достижимости
        for i, a1 in enumerate(phi):
            for j, a2 in enumerate(psi):
                l = (np.cos(a1) * np.cos(a2) * np.eye(param['dim'])[axis[0]] +
                     np.sin(a1) * np.cos(a2) * np.eye(param['dim'])[axis[1]] +
                     np.sin(a2) * np.eye(param['dim'])[axis[2]])

                # Для аппроксимации используем внутренние оценки
                c_int, Q_int = self.get_int_ellipse(param, t, l)
                x_reach[i, j], y_reach[i, j], z_reach[i, j] = self.support_func(c_int, Q_int, l)[axis]

        return x_reach, y_reach, z_reach

    def calculate_all_projections(self, param, t, n=15, axis=np.array([0, 1, 2])):
        """Вычисление всех проекций один раз"""
        print("Вычисление проекций...")

        # Вычисляем внешние оценки
        print("Внешние оценки...")
        self.proj_ext = self.get_3d_proj_single(param, t, n, axis, 'ext')

        # Вычисляем внутренние оценки
        print("Внутренние оценки...")
        self.proj_int = self.get_3d_proj_single(param, t, n, axis, 'int')

        # Вычисляем аппроксимацию множества достижимости
        print("Множество достижимости...")
        self.proj_reachable = self.get_reachable_set_approximation(param, t, n, axis)

        print("Все проекции вычислены")

    def get_3d_proj_single(self, param, t, n=15, axis=np.array([0, 1, 2]), mode='ext'):
        """Вычисление одной проекции"""
        phi = np.linspace(0, 2 * np.pi, 2 * n)
        psi = np.linspace(-np.pi / 2, np.pi / 2, n)
        x = np.zeros((2 * n, n))
        y = np.zeros((2 * n, n))
        z = np.zeros((2 * n, n))

        for i, a1 in enumerate(phi):
            for j, a2 in enumerate(psi):
                l = (np.cos(a1) * np.cos(a2) * np.eye(param['dim'])[axis[0]] +
                     np.sin(a1) * np.cos(a2) * np.eye(param['dim'])[axis[1]] +
                     np.sin(a2) * np.eye(param['dim'])[axis[2]])

                if mode == 'ext':
                    c_ext, Q_ext = self.get_ext_ellipse(param, t, l)
                    support_pt = self.support_func(c_ext, Q_ext, l)
                    x[i, j], y[i, j], z[i, j] = support_pt[axis]
                elif mode == 'int':
                    c_int, Q_int = self.get_int_ellipse(param, t, l)
                    support_pt = self.support_func(c_int, Q_int, l)
                    x[i, j], y[i, j], z[i, j] = support_pt[axis]

        return (x, y, z)

    def plot_3d_estimates(self, param, t=1.0):
        self.param = param
        self.current_t = t

        # Вычисляем все проекции один раз
        self.calculate_all_projections(param, t)

        # Создаем график с кнопками
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Область для чекбоксов
        plt.subplots_adjust(bottom=0.2)

        # Создаем чекбоксы для выбора отображаемых элементов
        ax_check = plt.axes([0.1, 0.02, 0.25, 0.12])
        self.check_buttons = CheckButtons(ax_check,
                                          ['Множество достижимости', 'Внутренние оценки', 'Внешние оценки'],
                                          [True, True, True])

        self.check_buttons.on_clicked(self.toggle_display)

        # Первоначальная отрисовка
        self.update_plot()
        plt.show()

    def toggle_display(self, label):
        """Обработчик переключения чекбоксов"""
        if label == 'Множество достижимости':
            self.display_states['reachable'] = not self.display_states['reachable']
        elif label == 'Внутренние оценки':
            self.display_states['internal'] = not self.display_states['internal']
        elif label == 'Внешние оценки':
            self.display_states['external'] = not self.display_states['external']

        self.update_plot()

    def update_plot(self, event=None):
        self.ax.clear()

        # Отрисовка в зависимости от выбранных чекбоксов
        if self.display_states['reachable'] and self.proj_reachable is not None:
            self.ax.plot_surface(self.proj_reachable[0], self.proj_reachable[1], self.proj_reachable[2],
                                 alpha=1, color='green', label='Множество достижимости')

        if self.display_states['internal'] and self.proj_int is not None:
            self.ax.plot_surface(self.proj_int[0], self.proj_int[1], self.proj_int[2],
                                 alpha=1, color='blue', label='Внутренние оценки')

        if self.display_states['external'] and self.proj_ext is not None:
            self.ax.plot_surface(self.proj_ext[0], self.proj_ext[1], self.proj_ext[2],
                                 alpha=1, color='red', label='Внешние оценки')

        self.ax.set_xlabel('X1')
        self.ax.set_ylabel('X2')
        self.ax.set_zlabel('X3')

        # Создаем заголовок с информацией о выбранных опциях
        active_options = []
        if self.display_states['reachable']:
            active_options.append('множество достижимости')
        if self.display_states['internal']:
            active_options.append('внутренние оценки')
        if self.display_states['external']:
            active_options.append('внешние оценки')

        title = f'3D проекция (t={self.current_t})'
        if active_options:
            title += f'\nПоказано: {", ".join(active_options)}'

        self.ax.set_title(title)
        self.ax.legend()
        plt.draw()


def get_user_matrix_input(dim, matrix_name):
    """Получение матрицы от пользователя через диалоговые окна"""
    matrix = np.zeros((dim, dim))
    root = tk.Tk()
    root.withdraw()

    for i in range(dim):
        for j in range(dim):
            value = simpledialog.askfloat(
                f"Матрица {matrix_name}",
                f"Введите элемент [{i + 1},{j + 1}]:",
                initialvalue=1.0 if i == j else 0.0
            )
            if value is None:
                return None
            matrix[i, j] = value

    return matrix


def create_example_1():
    """Пример 1: Простая система"""
    A = lambda t: np.array([[1, 0.1 * t, 0],
                            [0, -1.5, 0.2],
                            [0.1, 0, -1]])
    B = lambda t: np.array([[1, 0.3, 0],
                            [0, 1 + 0.1 * t, 0],
                            [0, 0, 1]])
    q = lambda t: np.zeros(3)
    Q = lambda t: np.eye(3) * (1 + 0.1 * t)

    return {
        'A': A, 'B': B, 'q': q, 'Q': Q,
        't0': 0, 'x0': np.zeros(3), 'X0': np.eye(3),
        'gr_sz': 500, 'dim': 3
    }


def create_example_2():
    """Пример 2: Система с экспоненциальными коэффициентами"""
    A = lambda t: np.eye(3) * (t + 1)
    B = lambda t: np.array([[1, 2, 3],
                            [2, 0, -0.1 * t],
                            [2, 2, 1]])
    q = lambda t: np.zeros(3)
    Q = lambda t: np.eye(3) * np.exp(0.1 * t)

    return {
        'A': A, 'B': B, 'q': q, 'Q': Q,
        't0': 0, 'x0': np.zeros(3), 'X0': np.eye(3),
        'gr_sz': 500, 'dim': 3
    }


def create_example_3():
    """Пример 3: Осциллирующая система"""
    A = lambda t: np.array([[0, 1, 0],
                            [-1, -0.1, 0],
                            [0, 0, -0.5]])
    B = lambda t: np.array([[np.cos(0.1 * t), 0, 0],
                            [0, np.sin(0.1 * t), 0],
                            [0, 0, 1]])
    q = lambda t: np.zeros(3)
    Q = lambda t: np.eye(3) * (1 + 0.05 * np.sin(t))

    return {
        'A': A, 'B': B, 'q': q, 'Q': Q,
        't0': 0, 'x0': np.zeros(3), 'X0': np.eye(3),
        'gr_sz': 500, 'dim': 3
    }


# Создаем экземпляр estimator
estimator = EllipsoidalEstimator3D()

# Выбор режима работы
root = tk.Tk()
root.withdraw()

choice = messagebox.askquestion("Выбор режима",
                                "Использовать готовый пример?\n\n"
                                "Да - выбрать из готовых примеров\n"
                                "Нет - ввести матрицы вручную")

if choice == 'yes':
    # Выбор примера
    example_choice = simpledialog.askinteger(
        "Выбор примера",
        "Выберите пример:\n"
        "1 - Простая система\n"
        "2 - Система с экспоненциальными коэффициентами\n"
        "3 - Осциллирующая система",
        minvalue=1, maxvalue=3, initialvalue=1
    )

    if example_choice == 1:
        param = create_example_1()
    elif example_choice == 2:
        param = create_example_2()
    elif example_choice == 3:
        param = create_example_3()
    else:
        param = create_example_1()

else:
    # Ручной ввод
    dim = simpledialog.askinteger("Размерность", "Введите размерность системы:",
                                  minvalue=2, maxvalue=10, initialvalue=3)
    if dim is None:
        exit()

    # Ввод матрицы A
    messagebox.showinfo("Матрица A", f"Введите элементы матрицы A ({dim}x{dim})")
    A_matrix = get_user_matrix_input(dim, "A")
    if A_matrix is None:
        exit()

    # Ввод матрицы B
    messagebox.showinfo("Матрица B", f"Введите элементы матрицы B ({dim}x{dim})")
    B_matrix = get_user_matrix_input(dim, "B")
    if B_matrix is None:
        exit()

    # Создаем функции для матриц
    A = lambda t: A_matrix
    B = lambda t: B_matrix
    q = lambda t: np.zeros(dim)
    Q = lambda t: np.eye(dim)

    param = {
        'A': A, 'B': B, 'q': q, 'Q': Q,
        't0': 0, 'x0': np.zeros(dim), 'X0': np.eye(dim),
        'gr_sz': 500, 'dim': dim
    }

# Ввод времени
t = simpledialog.askfloat("Время", "Введите время t для анализа:",
                          minvalue=0.1, maxvalue=10.0, initialvalue=1.0)
if t is None:
    t = 1.0

# Запуск визуализации
estimator.plot_3d_estimates(param, t)