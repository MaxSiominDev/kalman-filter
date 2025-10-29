"""
Фрагмент 3: Прогоняем фильтр на реальных данных
Генерируем движение с шумом и смотрим, как фильтр справляется
"""

import numpy as np
from fragment_1 import KalmanFilter1D

# Берем те же параметры, что настроили раньше
dt = 1.0
F = np.array([[1.0, dt], [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
Q = np.array([[0.001, 0.0], [0.0, 0.001]])
R = np.array([[0.05]])
x0 = np.array([[0.0], [1.0]])
P0 = np.eye(2) * 1.0

# Создаем фильтр
kf = KalmanFilter1D(F, H, Q, R, x0, P0)

# ============================================
# Симулируем движение объекта
# ============================================

np.random.seed(69)  # чтобы результаты повторялись

# Сколько точек генерим
n_steps = 50
true_positions = []
true_velocities = []
noisy_measurements = []

# Реальное движение
x_true = 0.0  # стартовая позиция
v_true = 1.0  # едет с постоянной скоростью

print("Генерим тестовые данные...")
print(f"Точек: {n_steps}")
print(f"Скорость объекта: {v_true} м/с")
print()

for step in range(n_steps):
    # Немного случайных помех (ветер, неровности дороги итп)
    process_noise = np.random.normal(0, np.sqrt(0.001))

    # Обновляем реальную позицию
    x_true += v_true * dt + process_noise

    # Запоминаем правду
    true_positions.append(x_true)
    true_velocities.append(v_true)

    # Датчик врет (GPS неточен, например)
    measurement_noise = np.random.normal(0, np.sqrt(0.05))
    z_noisy = x_true + measurement_noise
    noisy_measurements.append(z_noisy)

# Делаем массивы для удобства
true_positions = np.array(true_positions)
true_velocities = np.array(true_velocities)
noisy_measurements = np.array(noisy_measurements)

print(f"✓ Сгенерили {n_steps} измерений")
print()

# ============================================
# Запускаем фильтр
# ============================================

print("Запускаем фильтр Калмана...")

estimated_positions = []
estimated_velocities = []
position_uncertainties = []

for step, z_meas in enumerate(noisy_measurements):
    # Упаковываем измерение
    z = np.array([[z_meas]])

    # Делаем шаг фильтра (предсказание + коррекция)
    x_est = kf.step(z)

    # Сохраняем оценки
    estimated_positions.append(x_est[0, 0])
    estimated_velocities.append(x_est[1, 0])

    # Сохраняем насколько фильтр уверен в оценке
    P = kf.get_covariance()
    position_uncertainties.append(np.sqrt(P[0, 0]))

# Снова в массивы
estimated_positions = np.array(estimated_positions)
estimated_velocities = np.array(estimated_velocities)
position_uncertainties = np.array(position_uncertainties)

print(f"✓ Обработали все {n_steps} измерений")
print()

# ============================================
# Смотрим насколько хорошо сработало
# ============================================

print("\nРЕЗУЛЬТАТЫ\n")

# Ошибки сырых измерений
measurement_errors = np.abs(noisy_measurements - true_positions)
mae_measurements = np.mean(measurement_errors)
rmse_measurements = np.sqrt(np.mean(measurement_errors ** 2))

# Ошибки после фильтрации
estimation_errors = np.abs(estimated_positions - true_positions)
mae_estimation = np.mean(estimation_errors)
rmse_estimation = np.sqrt(np.mean(estimation_errors ** 2))

# Насколько точно угадали скорость
velocity_errors = np.abs(estimated_velocities - true_velocities)
mae_velocity = np.mean(velocity_errors)

print()
print("Ошибка определения позиции:")
print(f"  Сырые измерения с датчика:")
print(f"    MAE:  {mae_measurements:.4f} м")
print(f"    RMSE: {rmse_measurements:.4f} м")
print()
print(f"  После фильтра Калмана:")
print(f"    MAE:  {mae_estimation:.4f} м")
print(f"    RMSE: {rmse_estimation:.4f} м")
print()
print(f"  → Фильтр улучшил точность на {(1 - mae_estimation / mae_measurements) * 100:.1f}%")
print()
print("Оценка скорости:")
print(f"  MAE: {mae_velocity:.4f} м/с")
print(f"  (скорость мы вообще не измеряли, фильтр сам вычислил!)")
print()
print(f"Средняя неопределенность: ±{np.mean(position_uncertainties):.4f} м")
print()

# Проверяем адекватность фильтра
# Невязки должны быть случайными, без систематической ошибки
# (Если что невязка это diff/residual)
innovations = noisy_measurements - (H @ kf.F @ np.column_stack([estimated_positions, estimated_velocities]).T)[0, :]
print(f"Среднее невязок (должно быть близко к 0): {np.mean(innovations):.4f}")
print(f"Разброс невязок: {np.std(innovations):.4f}")
print()

print("=" * 60)
print("✓ Фильтр отработал нормально")
print("=" * 60)

# Сохраняем для графиков
if __name__ == "__main__":
    print()
    print("Теперь можно визуализировать в fragment_4.py")
