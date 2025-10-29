"""
Фрагмент 6: Тестируем OpenCV KalmanFilter
Показываем фильтр Калмана из OpenCV на примере 2D трекинга
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("OpenCV найден")
print(f"Версия: {cv2.__version__}")
print()

# ============================================
# Настраиваем фильтр OpenCV
# ============================================

# Отслеживаем точку в 2D
# Состояние: (x, y, vx, vy) — координаты и скорости
# Измеряем: (x, y) — только координаты
kalman = cv2.KalmanFilter(4, 2)

# Какие части состояния мы видим
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

dt = 1.0
# Как состояние меняется со временем (равномерное движение)
kalman.transitionMatrix = np.array([[1, 0, dt, 0],
                                    [0, 1, 0, dt],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

# Шум динамики
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

# Шум измерений
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0

# Стартуем из начала координат, движемся по диагонали
kalman.statePre = np.array([[0], [0], [1], [1]], np.float32)
kalman.statePost = np.array([[0], [0], [1], [1]], np.float32)
kalman.errorCovPost = np.eye(4, dtype=np.float32)

print("Настройки OpenCV Kalman:")
print(f"  Отслеживаем: 4 переменные (x, y, vx, vy)")
print(f"  Измеряем: 2 переменные (x, y)")
print(f"  Матрица движения:\n{kalman.transitionMatrix}")
print()

# ============================================
# Генерим 2D траекторию
# ============================================

np.random.seed(42)
n_points = 50

# Реальное движение, измерения, оценки
true_trajectory = []
measurements = []
estimated_trajectory = []

print("Симулируем движение объекта по плоскости...")

for i in range(n_points):
    # Реально объект движется по диагонали (чуть-чуть дрожит)
    true_x = i + np.random.normal(0, 0.1)
    true_y = i + np.random.normal(0, 0.1)
    true_trajectory.append((true_x, true_y))

    # Датчик видит с большой погрешностью (плохой GPS, например)
    meas_x = true_x + np.random.normal(0, 2.5)
    meas_y = true_y + np.random.normal(0, 2.5)
    measurement = np.array([[meas_x], [meas_y]], np.float32)
    measurements.append((meas_x, meas_y))

    # Калман делает шаг
    prediction = kalman.predict()
    estimated = kalman.correct(measurement)

    # Запоминаем отфильтрованную позицию
    estimated_trajectory.append((estimated[0, 0], estimated[1, 0]))

print(f"✓ Обработали {n_points} точек")
print()

# ============================================
# Считаем качество
# ============================================

true_trajectory = np.array(true_trajectory)
measurements = np.array(measurements)
estimated_trajectory = np.array(estimated_trajectory)

# Евклидовы расстояния до истины
meas_errors = np.sqrt(np.sum((measurements - true_trajectory) ** 2, axis=1))
est_errors = np.sqrt(np.sum((estimated_trajectory - true_trajectory) ** 2, axis=1))

print("=" * 60)
print("РЕЗУЛЬТАТЫ OpenCV (2D трекинг)")
print("=" * 60)
print(f"Датчик врет на:        {np.mean(meas_errors):.4f} м")
print(f"Фильтр ошибается на:   {np.mean(est_errors):.4f} м")
print(f"Выигрыш:               {(1 - np.mean(est_errors) / np.mean(meas_errors)) * 100:.1f}%")
print("=" * 60)
print()

# ============================================
# Рисуем траектории
# ============================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Слева: траектории в 2D
ax1.plot(true_trajectory[:, 0], true_trajectory[:, 1],
         'k-', label='Реальная траектория', linewidth=2.5)
ax1.scatter(measurements[:, 0], measurements[:, 1],
            c='red', s=30, alpha=0.5, label='Показания датчика')
ax1.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1],
         'g-', label='Оценка OpenCV', linewidth=2.5)
ax1.set_xlabel('X, м', fontsize=12)
ax1.set_ylabel('Y, м', fontsize=12)
ax1.set_title('2D трекинг объекта (OpenCV KalmanFilter)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# Справа: ошибки по времени
time_steps = np.arange(n_points)
ax2.plot(time_steps, meas_errors, 'r-', label='Ошибка датчика', linewidth=2)
ax2.plot(time_steps, est_errors, 'g-', label='Ошибка фильтра', linewidth=2)
ax2.axhline(y=np.mean(meas_errors), color='red',
            linestyle='--', alpha=0.5, label=f'Среднее (датчик): {np.mean(meas_errors):.2f}')
ax2.axhline(y=np.mean(est_errors), color='green',
            linestyle='--', alpha=0.5, label=f'Среднее (фильтр): {np.mean(est_errors):.2f}')
ax2.set_xlabel('Шаг', fontsize=12)
ax2.set_ylabel('Ошибка, м', fontsize=12)
ax2.set_title('Точность трекинга во времени', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('opencv_kalman_result.png', dpi=300)
print("График: 'opencv_kalman_result.png'")
plt.show()
