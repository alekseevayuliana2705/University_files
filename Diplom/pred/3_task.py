import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

# Настройки
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10


class TaxModel:
    def __init__(self, H=500):
        self.H = H

        # ФИКСИРУЕМ кривую Лоренца (среднее неравенство)
        self.curve_type = 'medium_inequality'

        # ФИКСИРУЕМ функцию зависти согласно вашему требованию
        self.phi_max = 0.9
        self.phi_min = 0.1  # φ_min = 0.9 - 0.8 = 0.1

    def phi_function(self, gamma):
        """Фиксированная функция 'зависти' согласно вашему требованию: φ(x) = 0.9 - 0.8/arctg(1000)*arctg(x)"""
        return self.phi_max - 0.8 / np.arctan(1000) * np.arctan(gamma)

    def lorenz_curve(self, x):
        """Фиксированная кривая Лоренца (среднее неравенство)"""
        return x ** 2  # G ≈ 0.33

    def calculate_gini(self):
        """Расчет индекса Джини для фиксированной кривой"""
        x = np.linspace(0, 1, 1000)
        y = self.lorenz_curve(x)
        area_under_lorenz = np.trapezoid(y, x)
        return 1 - 2 * area_under_lorenz

    def relative_income(self, x):
        """Относительный доход ξ(x) = dL/dx"""
        return 2 * x

    # === ПРЯМАЯ ЗАДАЧА: φ → налоговая система ===
    def solve_direct_problem(self):
        """Решаем прямую задачу: по φ находим налоговую систему"""

        print("=== ПРЯМАЯ ЗАДАЧА ===")
        print(f"Кривая Лоренца: L(x) = x² (Gini ≈ 0.33)")
        print(f"Функция зависти: φ(γ) = 0.9 - 0.8/arctg(1000)*arctg(γ)")
        print(f"φ_max = {self.phi_max}, φ_min = {self.phi_min}")

        def integrand(w):
            xi = self.relative_income(w)
            phi_val = self.phi_function(xi)
            return xi / (1 - phi_val)

        # Интегрируем для нахождения C
        w_points = np.linspace(0.001, 0.999, 1000)
        integral_values = [integrand(w) for w in w_points]
        integral = np.trapezoid(integral_values, w_points)
        C = 1 / integral

        print(f"Константа C (прямая задача): {C:.4f}")

        # Вычисляем налоговую систему для всех 10 децилей
        decile_positions = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        tax_system = {}

        print(f"\nНалоговые ставки по децилям:")
        print(f"{'Дециль':<8} {'γ':<8} {'φ(γ)':<8} {'ψ₀(γ)':<8} {'Налог, %':<12} {'Субсидия, %':<12}")
        print("-" * 70)

        for i, x in enumerate(decile_positions):
            xi = self.relative_income(x)
            phi_val = self.phi_function(xi)
            psi_0 = C * xi / (1 - phi_val)
            tax_rate = (1 - psi_0 / xi) * 100 if xi > 0 else 0
            subsidy_rate = -tax_rate if tax_rate < 0 else 0  # Положительная субсидия

            tax_system[f'D{i + 1}'] = {
                'gamma': xi,
                'phi': phi_val,
                'psi_0': psi_0,
                'tax_rate': tax_rate,
                'subsidy_rate': subsidy_rate,
                'position': x
            }

            print(f"{i + 1:<8} {xi:<8.3f} {phi_val:<8.3f} {psi_0:<8.3f} {tax_rate:<12.1f} {subsidy_rate:<12.1f}")

        return tax_system, C

    # === ОБРАТНАЯ ЗАДАЧА: налоговая система → φ ===
    def solve_inverse_problem(self, tax_system):
        """Решаем обратную задачу: по налоговой системе находим φ"""

        print("\n=== ОБРАТНАЯ ЗАДАЧА ===")
        print("Восстанавливаем φ по налоговой системе и кривой Лоренца")

        # Собираем данные для восстановления
        gamma_points = [data['gamma'] for data in tax_system.values()]
        psi_points = [data['psi_0'] for data in tax_system.values()]

        # Функция для нахождения оптимальной C из условия согласованности
        def find_optimal_C():
            """Находим C, обеспечивающую выполнение условия согласованности"""

            def consistency_condition(C):
                """Условие согласованности: ∫₀¹ ψ₀(ŷ'(x)) dx = 1"""
                x_points = np.linspace(0.001, 0.999, 1000)
                integral_values = []

                for x in x_points:
                    gamma = self.relative_income(x)

                    # Интерполируем ψ₀(γ) для данного gamma
                    if gamma < min(gamma_points):
                        psi_0 = psi_points[0]  # экстраполяция вниз
                    elif gamma > max(gamma_points):
                        psi_0 = psi_points[-1]  # экстраполяция вверх
                    else:
                        # Линейная интерполяция между ближайшими точками
                        for j in range(len(gamma_points) - 1):
                            if gamma_points[j] <= gamma <= gamma_points[j + 1]:
                                t = (gamma - gamma_points[j]) / (gamma_points[j + 1] - gamma_points[j])
                                psi_0 = psi_points[j] * (1 - t) + psi_points[j + 1] * t
                                break
                        else:
                            psi_0 = psi_points[-1]

                    # Восстанавливаем φ и проверяем ограничения
                    phi_reconstructed = 1 - C * gamma / psi_0

                    # Штраф за нарушение ограничений
                    penalty = 0
                    if phi_reconstructed < 0:
                        penalty += (0 - phi_reconstructed) ** 2
                    if phi_reconstructed > 1:
                        penalty += (phi_reconstructed - 1) ** 2

                    integral_values.append(psi_0 + penalty * 10)  # Увеличиваем штраф

                integral = np.trapezoid(integral_values, x_points)
                # Целевая функция: минимизировать |integral - 1|
                return abs(integral - 1)

            # Ищем C методом оптимизации
            result = minimize(consistency_condition, x0=0.5, bounds=[(0.1, 2.0)], method='L-BFGS-B')
            return result.x[0]

        C_opt = find_optimal_C()
        print(f"Константа C (обратная задача): {C_opt:.4f}")

        # Проверяем условие согласованности с найденной C
        x_points = np.linspace(0.001, 0.999, 1000)
        integral_values = []
        for x in x_points:
            gamma = self.relative_income(x)
            # Простая интерполяция ψ₀
            if gamma <= min(gamma_points):
                psi_0 = psi_points[0]
            elif gamma >= max(gamma_points):
                psi_0 = psi_points[-1]
            else:
                idx = np.searchsorted(gamma_points, gamma)
                t = (gamma - gamma_points[idx - 1]) / (gamma_points[idx] - gamma_points[idx - 1])
                psi_0 = psi_points[idx - 1] * (1 - t) + psi_points[idx] * t
            integral_values.append(psi_0)

        integral_check = np.trapezoid(integral_values, x_points)
        print(f"Проверка согласованности: ∫ψ₀(ŷ'(x))dx = {integral_check:.4f}")

        # Восстанавливаем функцию φ по формуле с найденной C
        phi_reconstructed_points = []

        print(f"\nВосстановленные значения φ:")
        print(f"{'Дециль':<8} {'γ':<8} {'ψ₀(γ)':<8} {'φ исх.':<8} {'φ восст.':<8} {'Ошибка':<8}")
        print("-" * 60)

        total_error = 0
        for i, (decile, data) in enumerate(tax_system.items()):
            gamma = data['gamma']
            psi_0 = data['psi_0']
            phi_original = data['phi']

            # ВОССТАНАВЛИВАЕМ φ по формуле с найденной C
            phi_reconstructed = 1 - C_opt * gamma / psi_0

            phi_reconstructed_points.append(phi_reconstructed)

            error = abs(phi_original - phi_reconstructed)
            total_error += error

            print(
                f"{i + 1:<8} {gamma:<8.3f} {psi_0:<8.3f} {phi_original:<8.3f} {phi_reconstructed:<8.3f} {error:<8.3f}")

        print(f"Средняя ошибка: {total_error / 10:.6f}")

        # Создаем интерполяционную функцию для восстановленной φ
        phi_reconstructed_func = CubicSpline(gamma_points, phi_reconstructed_points)

        # Строим непрерывную функцию на всем диапазоне
        gamma_range = np.linspace(min(gamma_points), max(gamma_points), 200)
        phi_reconstructed_continuous = [phi_reconstructed_func(g) for g in gamma_range]

        return gamma_range, phi_reconstructed_continuous, C_opt, phi_reconstructed_points

    def run_analysis(self):
        """Запускаем полный анализ"""

        # ПРЯМАЯ ЗАДАЧА
        tax_system, C_direct = self.solve_direct_problem()

        # ОБРАТНАЯ ЗАДАЧА - находим C из условия согласованности!
        gamma_range, phi_reconstructed, C_inverse, phi_reconstructed_points = self.solve_inverse_problem(tax_system)

        # Исходная функция φ
        phi_original = [self.phi_function(g) for g in gamma_range]

        # Сравнение
        mse = np.mean((np.array(phi_original) - np.array(phi_reconstructed)) ** 2)

        print(f"\n=== РЕЗУЛЬТАТЫ СРАВНЕНИЯ ===")
        print(f"Среднеквадратичная ошибка (MSE): {mse:.6f}")
        print(f"Константа C (прямая/обратная): {C_direct:.4f} / {C_inverse:.4f}")
        print(f"Разница в C: {abs(C_direct - C_inverse):.6f}")

        # ВИЗУАЛИЗАЦИЯ
        self.plot_results(gamma_range, phi_original, phi_reconstructed, tax_system, mse, phi_reconstructed_points, C_direct, C_inverse)

        return mse, C_direct, C_inverse

    def plot_results(self, gamma_range, phi_original, phi_reconstructed, tax_system, mse, phi_reconstructed_points, C_direct, C_inverse):
        """Строим графики результатов"""

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # График 1: Кривая Лоренца
        x_lorenz = np.linspace(0, 1, 100)
        y_lorenz = self.lorenz_curve(x_lorenz)
        gini = self.calculate_gini()

        ax1.plot(x_lorenz, y_lorenz, 'b-', linewidth=2, label=f'Кривая Лоренца\nGini = {gini:.3f}')
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Равенство')

        # Отмечаем децили
        decile_positions = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        for i, pos in enumerate(decile_positions):
            ax1.plot([pos, pos], [0, self.lorenz_curve(pos)], 'r:', alpha=0.3)
            if i % 2 == 0:  # Подписи только для четных децилей
                ax1.text(pos, self.lorenz_curve(pos) - 0.05, f'D{i + 1}',
                         ha='center', va='top', fontsize=8)

        ax1.set_xlabel('Доля населения')
        ax1.set_ylabel('Доля дохода')
        ax1.set_title('Кривая Лоренца: L(x) = x²')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # График 2: Налоговая система с положительными субсидиями
        decile_numbers = np.arange(1, 11)
        subsidy_rates = [tax_system[f'D{i}']['subsidy_rate'] for i in range(1, 11)]
        tax_rates_positive = [max(0, tax_system[f'D{i}']['tax_rate']) for i in range(1, 11)]

        # Столбцы для субсидий (положительные)
        bars_subsidies = ax2.bar(decile_numbers, subsidy_rates, color='green', alpha=0.7, label='Субсидии')
        # Столбцы для налогов (положительные)
        bars_taxes = ax2.bar(decile_numbers, tax_rates_positive, bottom=subsidy_rates, color='red', alpha=0.7,
                             label='Налоги')

        ax2.set_xlabel('Дециль')
        ax2.set_ylabel('Ставка, %')
        ax2.set_title('Налоговая система')
        ax2.set_xticks(decile_numbers)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Добавляем значения на столбцы
        for i, (subsidy, tax) in enumerate(zip(subsidy_rates, tax_rates_positive)):
            if subsidy > 0:
                ax2.text(i + 1, subsidy / 2, f'+{subsidy:.1f}%', ha='center', va='center',
                         fontsize=8, fontweight='bold', color='white')
            if tax > 0:
                ax2.text(i + 1, subsidy + tax / 2, f'{tax:.1f}%', ha='center', va='center',
                         fontsize=8, fontweight='bold', color='white')

        # График 3: Сравнение функций φ
        ax3.plot(gamma_range, phi_original, 'b-', linewidth=3,
                 label=f'Исходная φ(γ)', alpha=0.8)
        ax3.plot(gamma_range, phi_reconstructed, 'r--', linewidth=2,
                 label=f'Восстановленная φ(γ)', alpha=0.8)

        # Точки по децилям
        gamma_points = [tax_system[f'D{i}']['gamma'] for i in range(1, 11)]
        phi_original_points = [tax_system[f'D{i}']['phi'] for i in range(1, 11)]

        ax3.scatter(gamma_points, phi_original_points, color='blue', s=60, alpha=0.7, zorder=5,
                    label='Исходная (точки)')
        ax3.scatter(gamma_points, phi_reconstructed_points, color='red', s=40, alpha=0.7, zorder=5,
                    label='Восстановленная (точки)')

        ax3.set_xlabel('Относительный доход γ')
        ax3.set_ylabel('φ(γ)')
        ax3.set_title(f'Сравнение функций зависти\nMSE = {mse:.6f}\nC: {C_direct:.4f} → {C_inverse:.4f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)

        plt.tight_layout()
        plt.show()


# Запуск анализа
def main():
    print("АНАЛИЗ ПРЯМОЙ И ОБРАТНОЙ ЗАДАЧИ")
    print("=" * 70)
    print("1. Фиксируем кривую Лоренца: L(x) = x² (Gini ≈ 0.33)")
    print("2. Фиксируем функцию зависти: φ(γ) = 0.9 - 0.8/arctg(1000)*arctg(γ)")
    print("3. Прямая задача: φ + Лоренц → налоговая система")
    print("4. Обратная задача: налоговая система + Лоренц → φ (находим C!)")
    print("5. Сравниваем исходную и восстановленную φ")
    print("=" * 70)

    model = TaxModel(H=500)
    mse, C_direct, C_inverse = model.run_analysis()

    print(f"\nФИНАЛЬНЫЙ ВЫВОД:")
    print(f"Функция зависти восстановлена с MSE = {mse:.6f}")
    print(f"Константы C: прямая = {C_direct:.4f}, обратная = {C_inverse:.4f}")

    if mse < 0.001 and abs(C_direct - C_inverse) < 0.01:
        print("✓ Восстановление ОТЛИЧНОЕ - модель работает корректно!")
    elif mse < 0.01:
        print("✓ Восстановление хорошее - небольшие различия")
    else:
        print("✗ Восстановление требует улучшения")


if __name__ == "__main__":
    main()