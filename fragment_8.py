"""
Фрагмент 8: Эксперименты с параметром R
Что будет если неправильно оценить шум измерений?
"""

import numpy as np
import matplotlib.pyplot as plt
from fragment_1 import KalmanFilter1D

print("\nЭКСПЕРИМЕНТ: как параметр R влияет на фильтр\n\n")

# ============================================
# Генерим одни и те же данные для всех тестов
# ============================================

np.random.seed(42)
n_steps = 50
dt = 1.0

# Реальное движение
true_positions = []
x_true = 0.0
v_true = 1.0

for i in range(n_steps):
    x_true += v_true * dt + np.random.normal(0, np.sqrt(0.001))
    true_positions.append(x_true)

# Датчик врет с фиксированным шумом
true_measurement_noise = 5.0
noisy_measurements = [x + np.random.normal(0, np.sqrt(true_measurement_noise))
                      for x in true_positions]

true_positions = np.array(true_positions)
noisy_measurements = np.array(noisy_measurements)

print(f"Данных: {n_steps} точек")
print(f"Реальный шум датчика: σ = {np.sqrt(true_measurement_noise):.3f}")
print()

# ============================================
# Пробуем разные R
# ============================================

R_values = [0.01, 0.5, 5.0, 50.0, 200.0]
results = {}

print("Что будем тестировать:")
for r_val in R_values:
    print(f"  R = {r_val:.3f} (σ = {np.sqrt(r_val):.3f})")
print()

# Всё остальное фиксировано
F = np.array([[1.0, dt], [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
Q = np.array([[1.0, 0.0], [0.0, 0.1]])
x0 = np.array([[0.0], [1.0]])
P0 = np.eye(2) * 10.0

print("Запускаем тесты...")

for r_val in R_values:
    R = np.array([[r_val]])

    # Новый фильтр с новым R
    kf = KalmanFilter1D(F, H, Q, R, x0, P0)

    # Прогоняем данные
    estimated_positions = []
    estimated_velocities = []
    kalman_gains = []

    for z_meas in noisy_measurements:
        kf.predict()

        # Запоминаем коэффициент усиления
        S = H @ kf.P @ H.T + R
        K = kf.P @ H.T @ np.linalg.inv(S)
        kalman_gains.append(K[0, 0])

        kf.update(np.array([[z_meas]]))
        estimated_positions.append(kf.x[0, 0])
        estimated_velocities.append(kf.x[1, 0])

    estimated_positions = np.array(estimated_positions)
    estimated_velocities = np.array(estimated_velocities)
    kalman_gains = np.array(kalman_gains)

    # Считаем ошибки
    errors = np.abs(estimated_positions - true_positions)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    # Сохраняем всё
    results[r_val] = {
        'positions': estimated_positions,
        'velocities': estimated_velocities,
        'errors': errors,
        'mae': mae,
        'rmse': rmse,
        'kalman_gains': kalman_gains,
        'final_velocity': estimated_velocities[-1]
    }

    print(f"  R = {r_val:.3f}: MAE = {mae:.4f}, RMSE = {rmse:.4f}, "
          f"K = {np.mean(kalman_gains):.3f}, v_конец = {estimated_velocities[-1]:.3f}")

print()
print("✓ Все тесты готовы")
print()

# ============================================
# Выводы
# ============================================

print("=" * 60)
print("ЧТО ПОЛУЧИЛОСЬ")
print("=" * 60)

# Какой R ближе к истине
optimal_r = min(R_values, key=lambda r: abs(r - true_measurement_noise))
print(f"Реальный шум R = {true_measurement_noise:.3f}")
print(f"Ближайший из тестов R = {optimal_r:.3f}")
print()

print("Как R влияет на работу:")
print()
print("  Маленький R (думаем что датчик точный):")
print(f"    R = {R_values[0]:.3f}: MAE = {results[R_values[0]]['mae']:.4f}")
print(f"    → фильтр верит измерениям, следует за шумом")
print()
print("  Правильный R:")
print(f"    R = {optimal_r:.3f}: MAE = {results[optimal_r]['mae']:.4f}")
print(f"    → баланс, лучшая точность")
print()
print("  Большой R (думаем что датчик врет):")
print(f"    R = {R_values[-1]:.3f}: MAE = {results[R_values[-1]]['mae']:.4f}")
print(f"    → фильтр игнорит измерения, верит модели")
print()
print("=" * 60)
print()

# ============================================
# Рисуем всё
# ============================================

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Траектории при разных R
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(range(n_steps), true_positions, 'k-',
         label='Реальность', linewidth=2.5, alpha=0.8)
ax1.scatter(range(n_steps), noisy_measurements,
            c='gray', s=20, alpha=0.3, label='Датчик')

colors = ['blue', 'cyan', 'green', 'orange', 'red']
for r_val, color in zip(R_values, colors):
    ax1.plot(range(n_steps), results[r_val]['positions'],
             color=color, linewidth=2, alpha=0.7,
             label=f'R={r_val:.2f} (err={results[r_val]["mae"]:.2f})')

ax1.set_xlabel('Шаг', fontsize=12)
ax1.set_ylabel('Позиция, м', fontsize=12)
ax1.set_title('Как R меняет оценку положения', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, ncol=3)
ax1.grid(True, alpha=0.3)

# 2. Ошибка от R
ax2 = fig.add_subplot(gs[1, 0])
maes = [results[r]['mae'] for r in R_values]
ax2.plot(R_values, maes, 'bo-', linewidth=2, markersize=8)
ax2.axvline(x=true_measurement_noise, color='red', linestyle='--',
            label=f'Истинный R={true_measurement_noise:.3f}')
ax2.set_xlabel('R (насколько врет датчик)', fontsize=11)
ax2.set_ylabel('MAE (ошибка)', fontsize=11)
ax2.set_title('Точность фильтра vs R', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

# 3. Коэффициент усиления K
ax3 = fig.add_subplot(gs[1, 1])
for r_val, color in zip(R_values, colors):
    ax3.plot(range(n_steps), results[r_val]['kalman_gains'],
             color=color, linewidth=1.5, alpha=0.7, label=f'R={r_val:.2f}')
ax3.set_xlabel('Шаг', fontsize=11)
ax3.set_ylabel('K (усиление)', fontsize=11)
ax3.set_title('Насколько фильтр верит измерениям', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Ошибки по времени
ax4 = fig.add_subplot(gs[2, 0])
for r_val, color in zip(R_values, colors):
    ax4.plot(range(n_steps), results[r_val]['errors'],
             color=color, linewidth=1.5, alpha=0.7, label=f'R={r_val:.2f}')
ax4.set_xlabel('Шаг', fontsize=11)
ax4.set_ylabel('Ошибка, м', fontsize=11)
ax4.set_title('Ошибка во времени', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# 5. Оценка скорости
ax5 = fig.add_subplot(gs[2, 1])
for r_val, color in zip(R_values, colors):
    ax5.plot(range(n_steps), results[r_val]['velocities'],
             color=color, linewidth=1.5, alpha=0.7, label=f'R={r_val:.2f}')
ax5.axhline(y=v_true, color='black', linestyle='--', linewidth=2, label='Настоящая')
ax5.set_xlabel('Шаг', fontsize=11)
ax5.set_ylabel('Скорость, м/с', fontsize=11)
ax5.set_title('Скорость (её не измеряем!)', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

plt.savefig('experiment_R_influence.png', dpi=300, bbox_inches='tight')
print("✓ Картинка: 'experiment_R_influence.png'")
plt.show()

print()
print("ИТОГО:")
print("  • Малый R → фильтр нервный, дергается вслед за шумом")
print("  • Правильный R → точность максимальна")
print("  • Большой R → фильтр ленивый, игнорит данные")
print("  • Усиление K падает со временем (растет уверенность)")
print("  • Скорость оценивается сама, хотя не измеряется!")
