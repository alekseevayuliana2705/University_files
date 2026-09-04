import numpy as np
import matplotlib.pyplot as plt

# Настройки
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 8  # Уменьшаем размер шрифта


class SimpleTaxModel:
    def __init__(self, H=500):
        self.H = H

    def phi_function(self, gamma, phi_max, phi_min):
        """Функция 'зависти' - чем меньше phi_min, тем сильнее зависть"""
        return phi_max - (phi_max - phi_min) / np.arctan(2 * self.H) * np.arctan(gamma)

    def lorenz_curve(self, x, curve_type):
        """Три целевые кривые Лоренца"""
        if curve_type == 'low_inequality':
            return x ** 1.5  # Низкое неравенство (G ≈ 0.2)
        elif curve_type == 'medium_inequality':
            return x ** 2  # Среднее неравенство (G ≈ 0.33)
        else:  # high_inequality
            return (np.exp(3 * x) - 1) / (np.exp(3) - 1)  # Высокое неравенство (G ≈ 0.44)

    def calculate_gini(self, lorenz_func):
        """Простой расчет индекса Джини"""
        x = np.linspace(0, 1, 1000)
        y = lorenz_func(x)
        area_under_lorenz = np.trapz(y, x)
        return 1 - 2 * area_under_lorenz

    def relative_income(self, y, curve_type):
        """Относительный доход ξ(y) = dŷ/dx"""
        if curve_type == 'low_inequality':
            x = y ** (2 / 3)  # обратная функция для x^1.5
            return 1.5 * x ** 0.5
        elif curve_type == 'medium_inequality':
            x = np.sqrt(y)  # обратная функция для x^2
            return 2 * x
        else:  # high_inequality
            x = np.log(1 + y * (np.exp(3) - 1)) / 3
            return 3 * np.exp(3 * x) / (np.exp(3) - 1)

    def calculate_tax_rates(self, phi_max, phi_min, curve_type):
        """Расчет налоговых ставок для комбинации зависти и кривой Лоренца"""

        # 1. Вычисляем константу C
        def integrand(w):
            xi = self.relative_income(w, curve_type)
            phi_val = self.phi_function(xi, phi_max, phi_min)
            result = xi / (1 - phi_val)
            return result

        # Простое численное интегрирование
        w_points = np.linspace(0.001, 0.999, 100)
        integral_values = [integrand(w) for w in w_points]
        integral = np.trapz(integral_values, w_points)
        C = 1 / integral

        # ДИАГНОСТИКА
        print(f"\n--- ДИАГНОСТИКА: {curve_type}, phi_max={phi_max}, phi_min={phi_min} ---")
        print(f"Интеграл = {integral:.4f}, C = {C:.4f}")

        # Диагностика для ключевых точек
        test_points = [0.1, 0.5, 0.9]
        print("Ключевые точки:")
        for w in test_points:
            xi = self.relative_income(w, curve_type)
            phi_val = self.phi_function(xi, phi_max, phi_min)
            denominator = 1 - phi_val
            term = C / denominator
            I_R = term - 1
            print(
                f"  w={w:.1f}: ξ={xi:.3f}, φ={phi_val:.3f}, 1-φ={denominator:.3f}, C/(1-φ)={term:.3f}, налог={I_R * 100:.1f}%")

        # 2. Считаем ставки для децилей
        deciles = [0.1, 0.5, 0.9]  # 1-й, 5-й, 9-й децили
        tax_rates = []

        for decile in deciles:
            xi = self.relative_income(decile, curve_type)
            phi_val = self.phi_function(xi, phi_max, phi_min)
            denominator = 1 - phi_val
            term = C / denominator
            I_R = term - 1
            tax_rates.append(I_R * 100)  # в процентах

        return tax_rates, C



print("АНАЛИЗ ВЛИЯНИЯ 'ЗАВИСТИ' И ЦЕЛЕВОГО РАСПРЕДЕЛЕНИЯ НА НАЛОГИ")
print("=" * 60)

model = SimpleTaxModel(H=500)

# Четыре уровня "зависти"
envy_levels = {
    'very_weak': (0.7, 0.65),  # Очень слабая зависть
    'weak': (0.9, 0.89),  # Слабая зависть: k = 0.89
    'medium': (0.9, 0.4),  # Умеренная зависть: k = 0.44
    'strong': (0.9, 0.1)  # Сильная зависть: k = 0.11
}

# Три целевых распределения
curves = {
    'low_inequality': '',
    'medium_inequality': '',
    'high_inequality': ''
}

# Создаем 3x4 = 12 графиков
fig, axes = plt.subplots(3, 4, figsize=(20, 12))

# Графики кривых Лоренца (первый столбец)
x = np.linspace(0, 1, 100)
colors = ['green', 'blue', 'red']

for i, (curve_type, label) in enumerate(curves.items()):
    y = model.lorenz_curve(x, curve_type)
    gini = model.calculate_gini(lambda x: model.lorenz_curve(x, curve_type))
    axes[i, 0].plot(x, y, color=colors[i], linewidth=2,
                    label=f'{label} (G={gini:.2f})')
    axes[i, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Абсолютное равенство')
    axes[i, 0].set_xlabel('Доля населения', fontsize=9)
    axes[i, 0].set_ylabel('Доля дохода', fontsize=9)
    axes[i, 0].set_title(f'Кривая Лоренца: {label}', fontsize=10)
    axes[i, 0].legend(fontsize=8)
    axes[i, 0].grid(True, alpha=0.3)
    axes[i, 0].tick_params(labelsize=8)

# Анализ для каждой комбинации
results = {}

for envy_name, (phi_max, phi_min) in envy_levels.items():
    envy_results = {}

    for curve_type, curve_label in curves.items():
        # Расчет налоговых ставок
        tax_rates, C = model.calculate_tax_rates(phi_max, phi_min, curve_type)
        envy_results[curve_type] = {
            'tax_rates': tax_rates,
            'C': C,
            'k': phi_min / phi_max  # степень зависти
        }

    results[envy_name] = envy_results

# Названия децилей и уровней зависти
decile_names = ['Бедные\n(1-й дециль)', 'Средний класс\n(5-й дециль)', 'Богатые\n(9-й дециль)']
envy_titles = {
    'very_weak': 'Очень слабая зависть\n(φ_max=0.7, φ_min=0.65)',
    'weak': 'Сильное подражание\n(φ_max=0.9, φ_min=0.89)',
    'medium': 'Умеренное подражание\n(φ_max=0.9, φ_min=0.4)',
    'strong': 'Слабое подражание\n(φ_max=0.9, φ_min=0.1)'
}

curve_order = ['low_inequality', 'medium_inequality', 'high_inequality']
envy_order = ['very_weak', 'weak', 'medium', 'strong']

# Заполняем графики налоговых ставок (столбцы 2-5)
for row, curve_type in enumerate(curve_order):
    for col, envy_name in enumerate(envy_order):
        if col == 0:  # первый столбец уже занят кривыми Лоренца
            continue

        ax = axes[row, col]
        envy_data = results[envy_name]
        phi_max, phi_min = envy_levels[envy_name]

        tax_rates = envy_data[curve_type]['tax_rates']

        # Столбчатая диаграмма
        x_pos = np.arange(len(decile_names))
        bars = ax.bar(x_pos, tax_rates, color=colors[row], alpha=0.8)

        # Подписи на столбцах
        for bar, rate in zip(bars, tax_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{rate:.0f}%', ha='center', va='bottom' if rate > 0 else 'top',
                    fontweight='bold', fontsize=8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(decile_names, fontsize=8)
        ax.set_ylabel('Налоговая ставка, %', fontsize=9)

        curve_label = curves[curve_type].split('(')[0].strip()
        title = f'{curve_label}\n{envy_titles[envy_name]}'
        ax.set_title(title, fontsize=9)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=8)

plt.tight_layout(pad=2.0)  # Увеличиваем отступы между графиками
plt.show()

# Вывод результатов в таблицу
print("\nТАБЛИЦА РЕЗУЛЬТАТОВ:")
print("=" * 100)
print(
    f"{'Уровень зависти':<25} {'Кривая Лоренца':<25} {'Субсидия бедным':<15} {'Ставка сред.класс':<15} {'Налог богатых':<15}")
print("-" * 100)

for envy_name in envy_order:
    for curve_type, curve_label in curves.items():
        tax_rates = results[envy_name][curve_type]['tax_rates']
        print(
            f"{envy_titles[envy_name].split('(')[0].strip():<25} {curve_label:<25} {tax_rates[0]:<15.0f}% {tax_rates[1]:<15.0f}% {tax_rates[2]:<15.0f}%")
