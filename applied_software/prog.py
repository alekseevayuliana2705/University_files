import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 100

PURPLE = '#9b59b6'
PINK = "#e43ce7"
YELLOW = '#f1c40f'
GOLD = "#6F00F7"

def generate_data(n=50, center=(55.7558, 37.6173), delta=5, with_weights=False):
    np.random.seed(42)
    r = delta * np.sqrt(np.random.rand(n))
    theta = np.random.uniform(0, 2*np.pi, n)
    
    lat = center[0] + r * np.sin(theta)
    lon = center[1] + r * np.cos(theta)
    
    if with_weights:
        w = np.random.rand(n)
        w = w / w.sum()
        return pd.DataFrame({'lat': lat, 'lon': lon, 'weight': w})
    return pd.DataFrame({'lat': lat, 'lon': lon})

def optimize_euclidean(df, eps=1e-8):
    coords = df[['lat', 'lon']].values
    z = coords[:3].mean(axis=0)
    path = [z.copy()]
    
    while True:
        dist = np.linalg.norm(coords - z, axis=1)
        dist = np.maximum(dist, 1e-10)
        z_new = (coords.T @ (1/dist)) / (1/dist).sum()
        path.append(z_new.copy())
        
        if np.linalg.norm(z_new - z) < eps:
            break
        z = z_new
    
    return z_new, np.array(path)

def optimize_geodesic(df, eps=1e-6):
    coords = df[['lat', 'lon']].values
    z = coords[:3].mean(axis=0)
    path = [z.copy()]
    
    while True:
        dist = np.array([geodesic(z, (r[0], r[1])).km for r in coords])
        dist = np.maximum(dist, 1e-10)
        z_new = (coords.T @ (1/dist)) / (1/dist).sum()
        path.append(z_new.copy())
        
        if geodesic(z, z_new).km < eps:
            break
        z = z_new
    
    return z_new, np.array(path)

def optimize_weighted(df, eps=1e-8):
    coords = df[['lat', 'lon']].values
    weights = df['weight'].values
    z = coords[:3].mean(axis=0)
    path = [z.copy()]
    
    while True:
        dist = np.linalg.norm(coords - z, axis=1)
        dist = np.maximum(dist, 1e-10)
        z_new = (coords.T @ (weights/dist)) / (weights/dist).sum()
        path.append(z_new.copy())
        
        if np.linalg.norm(z_new - z) < eps:
            break
        z = z_new
    
    return z_new, np.array(path)

def plot_part1(df, path, result):
    plt.figure(figsize=(10, 8))
    plt.scatter(df['lat'], df['lon'], c=PURPLE, s=80, 
                edgecolor='white', linewidth=1, label='Клиенты', alpha=0.7)
    plt.plot(path[:,0], path[:,1], color=PINK, linewidth=1, 
             marker='o', markersize=2, label='Траектория поиска')
    plt.scatter(path[0,0], path[0,1], c=YELLOW, s=150, 
                edgecolor='white', linewidth=2, marker='s', label='Начальное приближение')
    plt.scatter(result[0], result[1], c=GOLD, s=350, 
                edgecolor='white', linewidth=2, marker='*', label='Оптимальное положение офиса')
    plt.xlabel('Широта', fontsize=12)
    plt.ylabel('Долгота', fontsize=12)
    plt.title('Часть I: Плоская модель (равные веса)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('part1.eps', format='eps', bbox_inches='tight')
    plt.savefig('part1.png', format='png', bbox_inches='tight')
    plt.show()

def plot_part2(df, path, result):
    plt.figure(figsize=(10, 8))
    plt.scatter(df['lat'], df['lon'], c=PURPLE, s=80, 
                edgecolor='white', linewidth=1, label='Клиенты', alpha=0.7)
    plt.plot(path[:,0], path[:,1], color=PINK, linewidth=1, 
             marker='o', markersize=2, label='Траектория поиска')
    plt.scatter(path[0,0], path[0,1], c=YELLOW, s=150, 
                edgecolor='white', linewidth=2, marker='s', label='Начальное приближение')
    plt.scatter(result[0], result[1], c=GOLD, s=350, 
                edgecolor='white', linewidth=2, marker='*', label='Оптимальное положение офиса')
    plt.xlabel('Широта', fontsize=12)
    plt.ylabel('Долгота', fontsize=12)
    plt.title('Часть II: Геодезическая модель', fontsize=14, fontweight='bold')
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('part2.eps', format='eps', bbox_inches='tight')
    plt.savefig('part2.png', format='png', bbox_inches='tight')
    plt.show()

def plot_part3(df, path, result):
    plt.figure(figsize=(10, 8))
    sizes = 50 + 200 * df['weight']
    scatter = plt.scatter(df['lat'], df['lon'], c=df['weight'], 
                         s=sizes, cmap='RdPu', edgecolor='white', 
                         linewidth=1, alpha=0.8, label='Клиенты')
    plt.colorbar(scatter, label='Вес клиента')
    plt.plot(path[:,0], path[:,1], color=PINK, linewidth=1, 
             marker='o', markersize=2, label='Траектория поиска')
    plt.scatter(path[0,0], path[0,1], c=YELLOW, s=150, 
                edgecolor='white', linewidth=2, marker='s', label='Начальное приближение')
    plt.scatter(result[0], result[1], c=GOLD, s=350, 
                edgecolor='white', linewidth=2, marker='*', label='Оптимальное положение офиса')
    plt.xlabel('Широта', fontsize=12)
    plt.ylabel('Долгота', fontsize=12)
    plt.title('Часть III: Плоская модель с приоритетами', fontsize=14, fontweight='bold')
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('part3.eps', format='eps', bbox_inches='tight')
    plt.savefig('part3.png', format='png', bbox_inches='tight')
    plt.show()

def plot_convergence(histories, labels):
    plt.figure(figsize=(10, 6))
    colors = [PURPLE, PINK, "#4150f1"]
    
    for hist, label, color in zip(histories, labels, colors):
        hist_norm = hist / hist[0]
        plt.plot(range(len(hist)), hist_norm, 
                label=label, color=color, linewidth=2, marker='o', markersize=4)
    
    plt.xlabel('Итерация', fontsize=12)
    plt.ylabel('Нормализованное значение функции', fontsize=12)
    plt.title('Сравнение скорости сходимости методов', fontsize=14, fontweight='bold')
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('convergence.eps', format='eps', bbox_inches='tight')
    plt.savefig('convergence.png', format='png', bbox_inches='tight')
    plt.show()

df1 = generate_data(n=50, delta=5)
df2 = generate_data(n=50, delta=30)
df3 = generate_data(n=50, delta=5, with_weights=True)

print("ЧАСТЬ 1: Плоская модель, равные веса")
res1, path1 = optimize_euclidean(df1)
coords1 = df1[['lat', 'lon']].values
final_dist1 = np.linalg.norm(coords1 - res1, axis=1)
func1 = final_dist1.mean() 
print(f"Оптимум: ({res1[0]:.4f}, {res1[1]:.4f})")
print(f"Значение функционала: {func1*111:.2f} км")
print()

print("ЧАСТЬ 2: Геодезическая модель")
res2, path2 = optimize_geodesic(df2)
coords2 = df2[['lat', 'lon']].values
final_dist2 = np.array([geodesic(res2, (r[0], r[1])).km for r in coords2])
func2 = final_dist2.mean()
print(f"Оптимум: ({res2[0]:.4f}, {res2[1]:.4f})")
print(f"Значение функционала: {func2:.2f} км")
print()

print("ЧАСТЬ 3: Плоская модель с весами")
res3, path3 = optimize_weighted(df3)
coords3 = df3[['lat', 'lon']].values
weights3 = df3['weight'].values
final_dist3 = np.linalg.norm(coords3 - res3, axis=1)
func3 = (weights3 * final_dist3).sum() 
print(f"Оптимум: ({res3[0]:.4f}, {res3[1]:.4f})")
print(f"Значение функционала: {func3*111:.2f} км")
print()

plot_part1(df1, path1, res1)
plot_part2(df2, path2, res2)
plot_part3(df3, path3, res3)