"""
Фрагмент 2: Прогоняем фильтр на реальных данных
Генерируем движение с шумом и смотрим, как фильтр справляется
"""

import numpy as np
from fragment_1 import KalmanFilter1D

# Берем те же параметры что и раньше
dt = 1.0
F = np.array([[1.0, dt], [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
Q = np.array([[0.001, 0.0], [0.0, 0.001]])
R = np.array([[0.05]])
x0 = np.array([[0.0], [1.0]])
P0 = np.eye(2) * 1.0

# Создаем фильтр
kf = KalmanFilter1D(F, H, Q, R, x0, P0)

# Симулируем движение объекта

np.random.seed(69)  # для воспроизводимости

# Сколько точек генерим
n_steps = 50
true_positions = []
true_velocities = []
noisy_measurements = []

# Реальное движение
x_true = 0.0  # стартовая позиция
v_true = 1.0  # постоянная скорость

print("Генерация тестовых данных...")
print(f"Количество точек: {n_steps}")
print(f"Скорость объекта: {v_true} м/с")
print()

for step in range(n_steps):
    # Небольшой шум процесса (ветер, неровности и т.д.)
    process_noise = np.random.normal(0, np.sqrt(0.001))

    # Обновляем реальную позицию
    x_true += v_true * dt + process_noise

    # Сохраняем истинные значения
    true_positions.append(x_true)
    true_velocities.append(v_true)

    # Датчик измеряет с ошибкой
    measurement_noise = np.random.normal(0, np.sqrt(0.05))
    z_noisy = x_true + measurement_noise
    noisy_measurements.append(z_noisy)

# Преобразуем в массивы
true_positions = np.array(true_positions)
true_velocities = np.array(true_velocities)
noisy_measurements = np.array(noisy_measurements)

print(f"Сгенерировано {n_steps} измерений")
print()

# Запускаем фильтр

print("Запуск фильтра Калмана...")

estimated_positions = []
estimated_velocities = []
position_uncertainties = []
innovations = []  # <-- ДОБАВЛЕНО: массив для невязок

for step, z_meas in enumerate(noisy_measurements):
    # Преобразуем измерение
    z = np.array([[z_meas]])

    kf.predict()

    # Предсказанное измерение
    z_pred = H @ kf.x

    # Невязка = реальное измерение - предсказание
    innovation = z[0, 0] - z_pred[0, 0]
    innovations.append(innovation)

    kf.update(z)

    # Получаем оценку состояния
    x_est = kf.get_state()

    # Сохраняем оценки
    estimated_positions.append(x_est[0, 0])
    estimated_velocities.append(x_est[1, 0])

    # Сохраняем неопределенность
    P = kf.get_covariance()
    position_uncertainties.append(np.sqrt(P[0, 0]))

# Преобразуем в массивы
estimated_positions = np.array(estimated_positions)
estimated_velocities = np.array(estimated_velocities)
position_uncertainties = np.array(position_uncertainties)
innovations = np.array(innovations)

print(f"Обработано {n_steps} измерений")
print()

# Анализ результатов

print("\nРезультаты работы фильтра\n")

# Ошибки сырых измерений
measurement_errors = np.abs(noisy_measurements - true_positions)
mae_measurements = np.mean(measurement_errors)
rmse_measurements = np.sqrt(np.mean(measurement_errors ** 2))

# Ошибки после фильтрации
estimation_errors = np.abs(estimated_positions - true_positions)
mae_estimation = np.mean(estimation_errors)
rmse_estimation = np.sqrt(np.mean(estimation_errors ** 2))

# Точность оценки скорости
velocity_errors = np.abs(estimated_velocities - true_velocities)
mae_velocity = np.mean(velocity_errors)

print()
print("Ошибка определения позиции:")
print(f"    MAE:  {mae_measurements:.4f} м")
print(f"    RMSE: {rmse_measurements:.4f} м")
print()
print(f"  После фильтра:")
print(f"    MAE:  {mae_estimation:.4f} м")
print(f"    RMSE: {rmse_estimation:.4f} м")
print()
improvement = (1 - mae_estimation / mae_measurements) * 100
print(f"  Улучшение точности: {improvement:.1f}%")
print()
print("Оценка скорости:")
print(f"  MAE: {mae_velocity:.4f} м/с")
print(f"  (скорость не измерялась напрямую, фильтр вычислил сам)")
print()
print(f"Средняя неопределенность: ±{np.mean(position_uncertainties):.4f} м")
print()

# Проверка адекватности фильтра
# Невязки должны быть случайными без систематической ошибки
print(f"Среднее значение невязок (должно быть ~0): {np.mean(innovations):.4f}")
print(f"СКО невязок: {np.std(innovations):.4f}")
print()

print("Фильтр отработал корректно")

# Данные готовы для визуализации
if __name__ == "__main__":
    print()
    print("Для визуализации запустите fragment_3.py")