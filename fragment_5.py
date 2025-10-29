"""
Фрагмент 5: Эксперименты с параметром R
Что будет если неправильно оценить шум измерений?
"""

import numpy as np
import matplotlib.pyplot as plt
from fragment_1 import KalmanFilter1D

print("\nЭксперимент: как параметр R влияет на фильтр\n")

# Генерация тестовых данных

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

# Датчик с фиксированным шумом
true_measurement_noise = 5.0
noisy_measurements = [x + np.random.normal(0, np.sqrt(true_measurement_noise))
                      for x in true_positions]

true_positions = np.array(true_positions)
noisy_measurements = np.array(noisy_measurements)

print(f"Количество точек: {n_steps}")
print(f"Реальный шум датчика: σ = {np.sqrt(true_measurement_noise):.3f}")
print()

# Тестируем разные значения R

R_values = [0.01, 0.5, 5.0, 50.0, 200.0]
results = {}

print("Тестируемые значения R:")
for r_val in R_values:
    print(f"  R = {r_val:.3f} (σ = {np.sqrt(r_val):.3f})")
print()

# Фиксированные параметры
F = np.array([[1.0, dt], [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
Q = np.array([[1.0, 0.0], [0.0, 0.1]])
x0 = np.array([[0.0], [1.0]])
P0 = np.eye(2) * 10.0

print("Запуск тестов...")

for r_val in R_values:
    R = np.array([[r_val]])

    # Создаем фильтр с текущим R
    kf = KalmanFilter1D(F, H, Q, R, x0, P0)

    # Обработка данных
    estimated_positions = []
    estimated_velocities = []
    kalman_gains = []

    for z_meas in noisy_measurements:
        kf.predict()

        # Сохраняем коэффициент усиления
        S = H @ kf.P @ H.T + R
        K = kf.P @ H.T @ np.linalg.inv(S)
        kalman_gains.append(K[0, 0])

        kf.update(np.array([[z_meas]]))
        estimated_positions.append(kf.x[0, 0])
        estimated_velocities.append(kf.x[1, 0])

    estimated_positions = np.array(estimated_positions)
    estimated_velocities = np.array(estimated_velocities)
    kalman_gains = np.array(kalman_gains)

    # Расчет ошибок
    errors = np.abs(estimated_positions - true_positions)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    # Сохранение результатов
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
print("Тесты завершены")
print()

# Анализ результатов

print("Результаты эксперимента")
# Поиск оптимального R
optimal_r = min(R_values, key=lambda r: abs(r - true_measurement_noise))
print(f"Реальный шум R = {true_measurement_noise:.3f}")
print(f"Ближайшее тестовое значение R = {optimal_r:.3f}")
print()

print("Влияние R на работу фильтра:")
print()
print("  Маленький R (предполагаем точный датчик):")
print(f"    R = {R_values[0]:.3f}: MAE = {results[R_values[0]]['mae']:.4f}")
print(f"    Фильтр доверяет измерениям, следует за шумом")
print()
print("  Правильный R:")
print(f"    R = {optimal_r:.3f}: MAE = {results[optimal_r]['mae']:.4f}")
print(f"    Оптимальный баланс, лучшая точность")
print()
print("  Большой R (предполагаем неточный датчик):")
print(f"    R = {R_values[-1]:.3f}: MAE = {results[R_values[-1]]['mae']:.4f}")
print(f"    Фильтр игнорирует измерения, полагается на модель")
print()

# Визуализация

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
ax1.set_title('Влияние R на оценку положения', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, ncol=3)
ax1.grid(True, alpha=0.3)


plt.savefig('experiment_R_influence.png', dpi=300, bbox_inches='tight')
print("График сохранен: 'experiment_R_influence.png'")
plt.show()
