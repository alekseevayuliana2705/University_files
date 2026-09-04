"""
Решение обратной задачи модели динамики распределения доходов
Восстановление функции подражательного поведения φ(γ) и коэффициента β
Период: 1980-2010
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, UnivariateSpline, interp1d
from scipy.optimize import minimize
import warnings
import os
warnings.filterwarnings('ignore')

# =============================================================================
# УКАЗЫВАЕМ ПУТЬ К ПАПКЕ С ДАННЫМИ
# =============================================================================
DATA_PATH = "/Users/ulianaalekseeva/Downloads/Germany/"
YEARS_RANGE = range(1980, 2011)  # 1980-2010 включительно

# =============================================================================
# 1. ЗАГРУЗКА ДАННЫХ С ПРАВИЛЬНОЙ СТРУКТУРОЙ
# =============================================================================

def load_data_correctly():
    """
    Загружает данные с учетом реальной структуры файлов
    """
    
    # Читаем файлы, пропуская первые строки с описанием
    try:
        # Post данные (после налогов)
        post_0_50 = pd.read_csv(os.path.join(DATA_PATH, 'Post_0_50.csv'), 
                                sep=';', skiprows=5, header=None)  # Пропускаем 5 строк описания
        
        post_50_100 = pd.read_csv(os.path.join(DATA_PATH, 'Post_50_100.csv'), 
                                  sep=';', skiprows=5, header=None)
        
        # Pre данные (до налогов)
        pre_0_50 = pd.read_csv(os.path.join(DATA_PATH, 'Pre_0_50.csv'), 
                               sep=';', skiprows=5, header=None)
        
        pre_50_100 = pd.read_csv(os.path.join(DATA_PATH, 'Pre_50_100.csv'), 
                                 sep=';', skiprows=5, header=None)
        
        print(f"Размеры файлов после пропуска заголовков:")
        print(f"  Post_0_50: {post_0_50.shape}")
        print(f"  Post_50_100: {post_50_100.shape}")
        print(f"  Pre_0_50: {pre_0_50.shape}")
        print(f"  Pre_50_100: {pre_50_100.shape}")
        
    except Exception as e:
        print(f"Ошибка при чтении файлов: {e}")
        return None, None
    
    # Создаем словари для хранения данных
    post_data = {}
    pre_data = {}
    
    # Функция для безопасного преобразования в число
    def safe_float_conversion(value):
        try:
            if pd.notna(value) and value != '':
                return float(str(value).replace(',', '.'))
            return None
        except:
            return None
    
    # Функция для безопасного преобразования года
    def safe_year_conversion(year_val):
        try:
            if pd.notna(year_val) and year_val != '':
                return int(float(str(year_val).replace(',', '.')))
            return None
        except:
            return None
    
    # Обрабатываем Post_0_50.csv - данные по нижним 50%
    print("\nОбработка Post_0_50.csv...")
    for idx, row in post_0_50.iterrows():
        if len(row) >= 3:
            percentile = str(row[0]).strip() if pd.notna(row[0]) else ""
            year = safe_year_conversion(row[1])
            value = safe_float_conversion(row[2])
            
            if year is not None and value is not None and 1980 <= year <= 2010:
                if year not in post_data:
                    post_data[year] = {}
                
                # Определяем дециль по названию перцентиля
                if 'p0p10' in percentile:
                    post_data[year]['p0p10'] = value
                elif 'p10p20' in percentile:
                    post_data[year]['p10p20'] = value
                elif 'p20p30' in percentile:
                    post_data[year]['p20p30'] = value
                elif 'p30p40' in percentile:
                    post_data[year]['p30p40'] = value
                elif 'p40p50' in percentile:
                    post_data[year]['p40p50'] = value
    
    # Обрабатываем Post_50_100.csv - данные по верхним 50%
    print("Обработка Post_50_100.csv...")
    for idx, row in post_50_100.iterrows():
        if len(row) >= 3:
            percentile = str(row[0]).strip() if pd.notna(row[0]) else ""
            year = safe_year_conversion(row[1])
            value = safe_float_conversion(row[2])
            
            if year is not None and value is not None and 1980 <= year <= 2010:
                if year not in post_data:
                    post_data[year] = {}
                
                if 'p50p60' in percentile:
                    post_data[year]['p50p60'] = value
                elif 'p60p70' in percentile:
                    post_data[year]['p60p70'] = value
                elif 'p70p80' in percentile:
                    post_data[year]['p70p80'] = value
                elif 'p80p90' in percentile:
                    post_data[year]['p80p90'] = value
                elif 'p90p100' in percentile:
                    post_data[year]['p90p100'] = value
    
    # Обрабатываем Pre_0_50.csv - данные по нижним 50% до налогов
    print("Обработка Pre_0_50.csv...")
    for idx, row in pre_0_50.iterrows():
        if len(row) >= 3:
            percentile = str(row[0]).strip() if pd.notna(row[0]) else ""
            year = safe_year_conversion(row[1])
            value = safe_float_conversion(row[2])
            
            if year is not None and value is not None and 1980 <= year <= 2010:
                if year not in pre_data:
                    pre_data[year] = {}
                
                if 'p40p50' in percentile:
                    pre_data[year]['p40p50'] = value
                elif 'p30p40' in percentile:
                    pre_data[year]['p30p40'] = value
                elif 'p20p30' in percentile:
                    pre_data[year]['p20p30'] = value
                elif 'p10p20' in percentile:
                    pre_data[year]['p10p20'] = value
                elif 'p0p10' in percentile:
                    pre_data[year]['p0p10'] = value
    
    # Обрабатываем Pre_50_100.csv - данные по верхним 50% до налогов
    print("Обработка Pre_50_100.csv...")
    for idx, row in pre_50_100.iterrows():
        if len(row) >= 3:
            percentile = str(row[0]).strip() if pd.notna(row[0]) else ""
            year = safe_year_conversion(row[1])
            value = safe_float_conversion(row[2])
            
            if year is not None and value is not None and 1980 <= year <= 2010:
                if year not in pre_data:
                    pre_data[year] = {}
                
                if 'p90p100' in percentile:
                    pre_data[year]['p90p100'] = value
                elif 'p80p90' in percentile:
                    pre_data[year]['p80p90'] = value
                elif 'p70p80' in percentile:
                    pre_data[year]['p70p80'] = value
                elif 'p60p70' in percentile:
                    pre_data[year]['p60p70'] = value
                elif 'p50p60' in percentile:
                    pre_data[year]['p50p60'] = value
    
    print(f"\nЗагружено {len(post_data)} лет Post данных")
    print(f"Годы Post: {sorted(post_data.keys())}")
    print(f"Загружено {len(pre_data)} лет Pre данных")
    print(f"Годы Pre: {sorted(pre_data.keys())}")
    
    # Проверяем наличие всех необходимых децилей для каждого года
    complete_years = []
    for year in sorted(post_data.keys()):
        required_cols = ['p0p10', 'p10p20', 'p20p30', 'p30p40', 'p40p50',
                         'p50p60', 'p60p70', 'p70p80', 'p80p90', 'p90p100']
        if all(col in post_data[year] for col in required_cols):
            complete_years.append(year)
    
    print(f"Годы с полными данными Post: {complete_years}")
    
    return post_data, pre_data

# =============================================================================
# 2. ПОСТРОЕНИЕ КРИВЫХ ЛОРЕНЦА
# =============================================================================

def build_lorenz_curves(post_data):
    """
    Строит кривые Лоренца из Post данных
    """
    lorenz_data = {}
    
    for year, data in post_data.items():
        # Проверяем наличие всех децилей
        required_cols = ['p0p10', 'p10p20', 'p20p30', 'p30p40', 'p40p50',
                         'p50p60', 'p60p70', 'p70p80', 'p80p90', 'p90p100']
        
        if all(col in data for col in required_cols):
            shares = []
            valid = True
            
            for col in required_cols:
                val = data[col]
                if val > 0:
                    shares.append(val)
                else:
                    valid = False
                    break
            
            if valid and len(shares) == 10:
                shares = np.array(shares)
                total = shares.sum()
                
                if total > 0:
                    shares = shares / total  # Нормируем
                    
                    # Ординаты кривой Лоренца (накопленные доли)
                    lorenz = np.concatenate([[0], np.cumsum(shares)])
                    
                    lorenz_data[year] = {
                        'shares': shares,
                        'lorenz': lorenz,
                        'x_points': np.linspace(0, 1, 11)
                    }
                    print(f"Год {year}: успешно построена кривая Лоренца")
    
    return lorenz_data

# =============================================================================
# 3. ПОСТРОЕНИЕ ДАННЫХ ДЛЯ АНАЛИЗА ПЕРЕРАСПРЕДЕЛЕНИЯ
# =============================================================================

def build_redistribution_data(post_data, pre_data):
    """
    Объединяет Pre и Post данные для анализа перераспределения
    """
    common_years = sorted(set(post_data.keys()) & set(pre_data.keys()))
    
    redist_list = []
    
    for year in common_years:
        row = {'year': year}
        
        # Добавляем Post данные
        for key, value in post_data[year].items():
            row[f'{key}_post'] = value
        
        # Добавляем Pre данные
        for key, value in pre_data[year].items():
            row[f'{key}_pre'] = value
        
        redist_list.append(row)
    
    redist_data = pd.DataFrame(redist_list)
    
    print(f"Общих лет для анализа перераспределения: {len(redist_data)}")
    if len(redist_data) > 0:
        print(f"Годы: {sorted(redist_data['year'].tolist())}")
    
    return redist_data

# =============================================================================
# 4. ФУНКЦИЯ ПЕРЕРАСПРЕДЕЛЕНИЯ ψ₀(γ)
# =============================================================================

def build_psi0_function(redist_data, year=None):
    """
    Строит функцию перераспределения ψ₀(γ) на основе данных Pre/Post
    """
    if redist_data.empty:
        def psi0_simple(gamma):
            return gamma
        return psi0_simple
    
    if year is not None and year in redist_data['year'].values:
        data_row = redist_data[redist_data['year'] == year].iloc[0]
    else:
        # Усредняем по всем годам
        data_row = redist_data.mean(numeric_only=True)
    
    # Собираем доли дохода до и после налогов
    deciles = []
    
    # Децили 0-9
    decile_names_post = ['p0p10_post', 'p10p20_post', 'p20p30_post', 'p30p40_post', 
                         'p40p50_post', 'p50p60_post', 'p60p70_post', 'p70p80_post', 
                         'p80p90_post', 'p90p100_post']
    
    decile_names_pre = ['p0p10_pre', 'p10p20_pre', 'p20p30_pre', 'p30p40_pre', 
                        'p40p50_pre', 'p50p60_pre', 'p60p70_pre', 'p70p80_pre', 
                        'p80p90_pre', 'p90p100_pre']
    
    for i in range(10):
        post_col = decile_names_post[i]
        pre_col = decile_names_pre[i]
        
        if post_col in data_row.index and pre_col in data_row.index:
            post_val = data_row[post_col]
            pre_val = data_row[pre_col]
            
            if not pd.isna(pre_val) and not pd.isna(post_val) and pre_val > 0:
                deciles.append({
                    'decile': i,
                    'gamma_pre': float(pre_val),
                    'gamma_post': float(post_val)
                })
    
    if len(deciles) < 2:
        print("Предупреждение: недостаточно данных для построения ψ₀(γ)")
        def psi0_simple(gamma):
            return gamma
        return psi0_simple
    
    deciles_df = pd.DataFrame(deciles)
    
    # Нормируем суммы к 1
    total_pre = deciles_df['gamma_pre'].sum()
    total_post = deciles_df['gamma_post'].sum()
    
    if total_pre <= 0 or total_post <= 0:
        def psi0_simple(gamma):
            return gamma
        return psi0_simple
    
    deciles_df['gamma_pre_norm'] = deciles_df['gamma_pre'] / total_pre
    deciles_df['gamma_post_norm'] = deciles_df['gamma_post'] / total_post
    
    # Сортируем по gamma_pre
    deciles_df = deciles_df.sort_values('gamma_pre_norm')
    
    # Строим функцию ψ₀(γ)
    gamma_vals = deciles_df['gamma_pre_norm'].values
    post_vals = deciles_df['gamma_post_norm'].values
    
    # Вес = насколько изменилась доля
    weights = post_vals / gamma_vals
    
    try:
        # Интерполяция
        weight_func = interp1d(gamma_vals, weights, kind='linear', 
                               fill_value='extrapolate', bounds_error=False)
        
        def psi0_weighted(g):
            base = np.asarray(g)
            w = weight_func(np.clip(base, gamma_vals.min(), gamma_vals.max()))
            result = base * w
            result = np.maximum(result, 0)
            return result
    except:
        def psi0_weighted(g):
            return np.asarray(g)
    
    return psi0_weighted

# =============================================================================
# 5. АППРОКСИМАЦИЯ КРИВЫХ ЛОРЕНЦА
# =============================================================================

def create_lorenz_splines(lorenz_data):
    """
    Создает гладкие монотонные сплайны для кривых Лоренца
    """
    splines = {}
    
    for year, data in lorenz_data.items():
        x = data['x_points']
        y = data['lorenz']
        
        # Монотонная интерполяция PCHIP
        spline = PchipInterpolator(x, y)
        
        # Генерируем плотную сетку
        x_dense = np.linspace(0, 1, 200)
        y_dense = spline(x_dense)
        
        # Вычисляем производную (плотность дохода)
        gamma_dense = np.gradient(y_dense, x_dense)
        gamma_dense = np.maximum(gamma_dense, 1e-6)
        
        splines[year] = {
            'spline': spline,
            'x_dense': x_dense,
            'y_dense': y_dense,
            'gamma_dense': gamma_dense
        }
    
    return splines

# =============================================================================
# 6. ПАРАМЕТРИЗАЦИЯ ФУНКЦИИ φ(γ)
# =============================================================================

def phi_logistic(gamma, beta, phi0, phi1, k, gamma0):
    """
    Логистическая функция для φ(γ)
    """
    phi = phi1 + (phi0 - phi1) / (1 + np.exp(k * (gamma - gamma0)))
    return phi

# =============================================================================
# 7. ВЫЧИСЛЕНИЕ ФУНКЦИОНАЛА G(x; φ)
# =============================================================================

def compute_G(x, spline_data, phi_func, psi0_func, x_grid, gamma_grid):
    """
    Вычисляет G(x; φ)
    """
    gamma_w = gamma_grid
    y_x = spline_data['spline'](x)
    
    integrand = (1 - phi_func(gamma_w)) * psi0_func(gamma_w)
    
    idx_x = np.searchsorted(x_grid, x)
    if idx_x >= len(integrand):
        idx_x = len(integrand) - 1
    
    integral_to_x = np.trapz(integrand[:idx_x+1], x_grid[:idx_x+1])
    integral_total = np.trapz(integrand, x_grid)
    
    if integral_total == 0:
        return 0
    
    G = integral_to_x - y_x * integral_total
    return G

# =============================================================================
# 8. ФУНКЦИОНАЛ НЕВЯЗКИ J(β, φ)
# =============================================================================

def create_objective_function(splines_by_year, years, r, psi0_func):
    """
    Создает функцию для минимизации J(β, φ)
    """
    years = sorted(years)
    if len(years) < 2:
        return lambda x: 1e10
    
    x_grid = splines_by_year[years[0]]['x_dense']
    
    dt_list = [years[i+1] - years[i] for i in range(len(years)-1)]
    
    def objective(params):
        beta = params[0]
        phi0, phi1, k, gamma0 = params[1:]
        
        # Проверяем ограничения
        if phi1 < beta or phi1 > 1 or phi0 < 1 or k <= 0:
            return 1e10
        
        def phi_func(g):
            return phi_logistic(g, beta, phi0, phi1, k, gamma0)
        
        total_error = 0
        
        for i in range(len(years)-1):
            year1 = years[i]
            year2 = years[i+1]
            dt = dt_list[i]
            
            if year1 not in splines_by_year or year2 not in splines_by_year:
                continue
                
            data1 = splines_by_year[year1]
            data2 = splines_by_year[year2]
            
            G_vals = np.array([compute_G(x, data1, phi_func, psi0_func, 
                                         x_grid, data1['gamma_dense']) 
                               for x in x_grid])
            
            dy_model = dt * (r / (1 - beta)) * G_vals
            dy_real = data2['y_dense'] - data1['y_dense']
            
            error = np.trapz((dy_real - dy_model)**2, x_grid)
            total_error += error
        
        return total_error
    
    return objective

# =============================================================================
# 9. ОПТИМИЗАЦИЯ
# =============================================================================

def solve_inverse_problem(lorenz_data, redist_data, r=0.03):
    """
    Решает обратную задачу: находит β и φ(γ)
    """
    splines = create_lorenz_splines(lorenz_data)
    
    available_years = sorted([y for y in splines.keys() if 1980 <= y <= 2010])
    print(f"\nДоступные годы для оптимизации: {available_years}")
    
    if len(available_years) < 2:
        print("Ошибка: недостаточно данных для решения обратной задачи")
        return None, None, None
    
    psi0_func = build_psi0_function(redist_data)
    objective = create_objective_function(splines, available_years, r, psi0_func)
    
    # Начальное приближение
    x0 = [0.5, 2.0, 0.8, 10.0, 0.5]
    
    bounds = [
        (0.1, 0.9),      # beta
        (1.01, 5.0),     # phi0
        (0.1, 1.0),      # phi1
        (1.0, 50.0),     # k
        (0.1, 0.9)       # gamma0
    ]
    
    print("\nЗапуск оптимизации...")
    
    result = minimize(objective, x0, bounds=bounds, 
                      method='L-BFGS-B', 
                      options={'maxiter': 1000, 'disp': True})
    
    if result.success:
        beta_opt = result.x[0]
        phi0_opt, phi1_opt, k_opt, gamma0_opt = result.x[1:]
        
        print(f"\nРезультаты оптимизации:")
        print(f"β = {beta_opt:.4f}")
        print(f"φ(0) = {phi0_opt:.4f}")
        print(f"φ(1) = {phi1_opt:.4f}")
        print(f"k = {k_opt:.4f}")
        print(f"γ₀ = {gamma0_opt:.4f}")
        print(f"Функционал J = {result.fun:.6f}")
        
        def phi_optimal(gamma):
            return phi_logistic(gamma, beta_opt, phi0_opt, phi1_opt, k_opt, gamma0_opt)
        
        return beta_opt, phi_optimal, result
    else:
        print("Оптимизация не сошлась")
        print(result.message)
        return None, None, result

# =============================================================================
# 10. ВИЗУАЛИЗАЦИЯ
# =============================================================================

def plot_results(lorenz_data, beta, phi_func):
    """
    Визуализация результатов
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Кривые Лоренца
    ax = axes[0, 0]
    years_to_plot = sorted(lorenz_data.keys())[::5]
    for year in years_to_plot[:5]:
        x = lorenz_data[year]['x_points']
        y = lorenz_data[year]['lorenz']
        ax.plot(x, y, marker='o', label=f'{year}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Равенство')
    ax.set_xlabel('Доля населения')
    ax.set_ylabel('Доля дохода')
    ax.set_title('Кривые Лоренца (1980-2010)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Функция φ(γ)
    ax = axes[0, 1]
    gamma_plot = np.linspace(0, 1, 100)
    phi_plot = phi_func(gamma_plot)
    ax.plot(gamma_plot, phi_plot, 'b-', linewidth=2)
    ax.axhline(y=beta, color='r', linestyle='--', label=f'β = {beta:.3f}')
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('γ (доля дохода)')
    ax.set_ylabel('φ(γ)')
    ax.set_title('Функция подражательного поведения φ(γ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Распределение γ
    ax = axes[0, 2]
    splines = create_lorenz_splines(lorenz_data)
    for year in years_to_plot[:3]:
        x_dense = splines[year]['x_dense']
        gamma_dense = splines[year]['gamma_dense']
        ax.plot(x_dense, gamma_dense, label=f'{year}')
    ax.set_xlabel('Ранг населения x')
    ax.set_ylabel('γ(x) = dy/dx')
    ax.set_title('Плотность распределения дохода')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Динамика доли беднейших
    ax = axes[1, 0]
    years = sorted(lorenz_data.keys())
    poor_share = [lorenz_data[y]['shares'][0] * 100 for y in years]
    ax.plot(years, poor_share, 'o-', color='green')
    ax.set_xlabel('Год')
    ax.set_ylabel('Доля дохода нижнего дециля (%)')
    ax.set_title('Динамика доли беднейших 10%')
    ax.grid(True, alpha=0.3)
    
    # 5. Динамика доли богатейших
    ax = axes[1, 1]
    rich_share = [lorenz_data[y]['shares'][9] * 100 for y in years]
    ax.plot(years, rich_share, 'o-', color='red')
    ax.set_xlabel('Год')
    ax.set_ylabel('Доля дохода верхнего дециля (%)')
    ax.set_title('Динамика доли богатейших 10%')
    ax.grid(True, alpha=0.3)
    
    # 6. Коэффициент Джини
    ax = axes[1, 2]
    gini = []
    for y in years:
        x = np.linspace(0, 1, 11)
        lorenz = lorenz_data[y]['lorenz']
        gini_val = 1 - 2 * np.trapz(lorenz, x)
        gini.append(gini_val)
    
    ax.plot(years, gini, 'o-', color='purple')
    ax.set_xlabel('Год')
    ax.set_ylabel('Коэффициент Джини')
    ax.set_title('Динамика неравенства')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('inverse_problem_results.png', dpi=150)
    plt.show()

# =============================================================================
# ОСНОВНАЯ ПРОГРАММА
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("РЕШЕНИЕ ОБРАТНОЙ ЗАДАЧИ МОДЕЛИ ДИНАМИКИ ДОХОДОВ")
    print(f"Период: 1980-2010")
    print("=" * 60)
    
    # Проверяем существование папки с данными
    if not os.path.exists(DATA_PATH):
        print(f"Ошибка: папка {DATA_PATH} не найдена!")
        print("Пожалуйста, проверьте путь к папке с данными.")
        exit(1)
    
    # 1. Загружаем данные с правильной структурой
    print("\n1. Загрузка данных...")
    post_data, pre_data = load_data_correctly()
    
    if not post_data or not pre_data:
        print("Ошибка: не удалось загрузить данные")
        exit(1)
    
    # 2. Строим кривые Лоренца
    print("\n2. Построение кривых Лоренца...")
    lorenz_data = build_lorenz_curves(post_data)
    print(f"Построено {len(lorenz_data)} кривых Лоренца")
    
    if not lorenz_data:
        print("Ошибка: не удалось построить кривые Лоренца")
        exit(1)
    
    # 3. Строим данные для анализа перераспределения
    print("\n3. Анализ перераспределения...")
    redist_data = build_redistribution_data(post_data, pre_data)
    
    # 4. Решаем обратную задачу
    print("\n4. Решение обратной задачи...")
    print("   Процентная ставка r = 0.03 (3%)")
    
    beta_opt, phi_opt, result = solve_inverse_problem(
        lorenz_data, redist_data, r=0.03
    )
    
    if beta_opt is not None:
        # 5. Визуализация
        print("\n5. Визуализация результатов...")
        plot_results(lorenz_data, beta_opt, phi_opt)
        
        # 6. Сохранение результатов
        print("\n6. Сохранение результатов...")
        
        gamma_save = np.linspace(0, 1, 100)
        phi_save = phi_opt(gamma_save)
        
        results_df = pd.DataFrame({
            'gamma': gamma_save,
            'phi': phi_save
        })
        results_df.to_csv('phi_function.csv', index=False)
        
        with open('parameters.txt', 'w') as f:
            f.write(f"beta = {beta_opt:.6f}\n")
            f.write(f"r = 0.03\n")
            f.write(f"J = {result.fun:.6f}\n")
            f.write(f"Период: 1980-2010\n")
        
        print("   Результаты сохранены в 'phi_function.csv' и 'parameters.txt'")
    
    print("\n" + "=" * 60)
    print("РАСЧЕТ ЗАВЕРШЕН")
    print("=" * 60)