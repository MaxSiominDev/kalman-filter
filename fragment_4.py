"""
Фрагмент 4: Сравниваем с библиотекой FilterPy
Проверяем что наша реализация работает как настоящая библиотека
"""

from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt

print("FilterPy найден, можно сравнивать")
print()

# Настраиваем FilterPy

# 2 переменных состояния (позиция, скорость)
# 1 измеряемая величина (позиция)
kf = KalmanFilter(dim_x=2, dim_z=1)

dt = 1.0

# Те же параметры что и у нас
kf.F = np.array([[1.0, dt],
                 [0.0, 1.0]])

kf.H = np.array([[1.0, 0.0]])

kf.Q = np.eye(2) * 0.001

kf.R = np.array([[0.05]])

kf.x = np.array([[0.0],
                 [1.0]])

kf.P = np.eye(2) * 1.0

print("Настройки FilterPy:")
print(f"  Матрица F:\n{kf.F}")
print(f"  Матрица H:\n{kf.H}")
print(f"  Шум Q:\n{kf.Q}")
print(f"  Шум R:\n{kf.R}")
print()

# Генерация данных

np.random.seed(42)
n_steps = 50

# Реальное движение
true_positions = []
x_true = 0.0
v_true = 1.0

for i in range(n_steps):
    x_true += v_true * dt + np.random.normal(0, np.sqrt(0.001))
    true_positions.append(x_true)

# Зашумленные измерения
noisy_measurements = [x + np.random.normal(0, np.sqrt(0.05))
                      for x in true_positions]

print(f"Данные готовы: {n_steps} точек")
print()

# Прогоняем через FilterPy

print("Запускаем FilterPy...")

filtered_positions = []
filtered_velocities = []

for z in noisy_measurements:
    # Предсказание + обновление
    kf.predict()
    kf.update([z])

    # Сохраняем результаты
    filtered_positions.append(kf.x[0, 0])
    filtered_velocities.append(kf.x[1, 0])

print(f"Обработка завершена")
print()

# Анализ результатов

true_positions = np.array(true_positions)
noisy_measurements = np.array(noisy_measurements)
filtered_positions = np.array(filtered_positions)

mae_meas = np.mean(np.abs(noisy_measurements - true_positions))
mae_filt = np.mean(np.abs(filtered_positions - true_positions))

print("Сравнение с FilterPy")
print(f"Ошибка датчика:       {mae_meas:.4f} м")
print(f"Ошибка FilterPy:      {mae_filt:.4f} м")
print(f"Улучшение:            {(1 - mae_filt / mae_meas) * 100:.1f}%")
print()

# Визуализация

plt.figure(figsize=(12, 6))
plt.plot(true_positions, 'k-', label='Реальная траектория', linewidth=2)
plt.scatter(range(n_steps), noisy_measurements,
            c='red', s=20, alpha=0.5, label='Зашумленные измерения')
plt.plot(filtered_positions, 'b-', label='FilterPy фильтр', linewidth=2)
plt.xlabel('Шаг')
plt.ylabel('Позиция, м')
plt.title('FilterPy - библиотечная реализация фильтра Калмана')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('filterpy_result.png', dpi=300)
print("График сохранен: 'filterpy_result.png'")
plt.show()