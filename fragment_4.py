"""
Фрагмент 4: Рисуем графики
Смотрим визуально, как фильтр справился с задачей
"""

import matplotlib.pyplot as plt
import numpy as np

# Запускаем всё заново, чтобы скрипт был самодостаточным
from fragment_1 import KalmanFilter1D

# Воспроизводим данные (те же параметры что раньше)
np.random.seed(42)
dt = 1.0
F = np.array([[1.0, dt], [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
Q = np.array([[0.001, 0.0], [0.0, 0.001]])
R = np.array([[0.05]])
x0 = np.array([[0.0], [1.0]])
P0 = np.eye(2)

kf = KalmanFilter1D(F, H, Q, R, x0, P0)

# Генерим те же данные
n_steps = 50
x_true = 0.0
v_true = 1.0
true_positions = []
noisy_measurements = []

for step in range(n_steps):
    process_noise = np.random.normal(0, np.sqrt(0.001))
    x_true += v_true * dt + process_noise
    true_positions.append(x_true)
    measurement_noise = np.random.normal(0, np.sqrt(0.05))
    noisy_measurements.append(x_true + measurement_noise)

# Фильтруем
estimated_positions = []
for z_meas in noisy_measurements:
    x_est = kf.step(np.array([[z_meas]]))
    estimated_positions.append(x_est[0, 0])

# ============================================
# РИСУЕМ
# ============================================

# Два графика друг под другом
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# ============================================
# Верхний график: траектории
# ============================================
ax1 = axes[0]

time_steps = np.arange(n_steps)

# Реальное движение (черная линия)
ax1.plot(time_steps, true_positions, 'k-',
         label='Реальная позиция', linewidth=2.5, alpha=0.8)

# Что показал датчик (красные точки)
ax1.scatter(time_steps, noisy_measurements,
            color='red', s=30, alpha=0.5, label='Измерения (с шумом)')

# Что выдал фильтр (зеленая линия)
ax1.plot(time_steps, estimated_positions, 'g-',
         label='Оценка фильтра', linewidth=2.5, alpha=0.9)

ax1.set_xlabel('Шаг времени', fontsize=13)
ax1.set_ylabel('Позиция, м', fontsize=13)
ax1.set_title('Фильтр Калмана в действии',
              fontsize=15, fontweight='bold')
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3)

# ============================================
# Нижний график: ошибки
# ============================================
ax2 = axes[1]

# Считаем ошибки
measurement_errors = np.abs(np.array(noisy_measurements) - np.array(true_positions))
estimation_errors = np.abs(np.array(estimated_positions) - np.array(true_positions))

# Ошибка датчика (красная)
ax2.plot(time_steps, measurement_errors, 'r-',
         label='Ошибка датчика', linewidth=2, alpha=0.7)

# Ошибка фильтра (зеленая)
ax2.plot(time_steps, estimation_errors, 'g-',
         label='Ошибка фильтра', linewidth=2, alpha=0.7)

# Средние значения (пунктир)
ax2.axhline(y=np.mean(measurement_errors), color='red',
            linestyle='--', linewidth=1.5, alpha=0.5,
            label=f'Средняя ошибка датчика: {np.mean(measurement_errors):.3f} м')
ax2.axhline(y=np.mean(estimation_errors), color='green',
            linestyle='--', linewidth=1.5, alpha=0.5,
            label=f'Средняя ошибка фильтра: {np.mean(estimation_errors):.3f} м')

ax2.set_xlabel('Шаг времени', fontsize=13)
ax2.set_ylabel('Ошибка, м', fontsize=13)
ax2.set_title('Насколько точно определили позицию',
              fontsize=15, fontweight='bold')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kalman_filter_result.png', dpi=300, bbox_inches='tight')
print("✓ Картинка сохранена: 'kalman_filter_result.png'")
plt.show()

# ============================================
# Выводим цифры
# ============================================
print("\n\nИТОГИ\n")
print(f"Датчик ошибается в среднем на:  {np.mean(measurement_errors):.4f} м")
print(f"Фильтр ошибается в среднем на:  {np.mean(estimation_errors):.4f} м")
print(f"Выигрыш от фильтрации:          {(1 - np.mean(estimation_errors)/np.mean(measurement_errors))*100:.1f}%")
print("=" * 60)
