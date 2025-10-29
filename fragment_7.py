"""
Фрагмент 7: Extended Kalman Filter (EKF)
Для нелинейных систем — линеаризуем через производные
"""

import numpy as np
import matplotlib.pyplot as plt


class ExtendedKalmanFilter:
    """
    Расширенный фильтр Калмана

    Для систем где:
    - динамика нелинейная: x_k = f(x_{k-1}) + шум
    - измерения нелинейные: z_k = h(x_k) + шум

    Фокус: линеаризуем f и h через якобианы (матрицы производных)
    """

    def __init__(self, f, h, F_jacobian, H_jacobian, Q, R, x0, P0):
        """
        f — функция динамики (как меняется состояние)
        h — функция измерения (что видим)
        F_jacobian — производные f по состоянию
        H_jacobian — производные h по состоянию
        Q, R, x0, P0 — как обычно
        """
        self.f = f
        self.h = h
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self, u=None):
        """Предсказание с нелинейной динамикой"""
        # Прогнозируем состояние нелинейной функцией
        self.x = self.f(self.x, u)

        # Линеаризуем вокруг текущей точки
        F = self.F_jacobian(self.x)

        # Обновляем ковариацию (используя линеаризацию)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """Коррекция с нелинейными измерениями"""
        # Линеаризуем функцию измерения
        H = self.H_jacobian(self.x)

        # Предсказываем что должны увидеть (нелинейно)
        z_pred = self.h(self.x)

        # Невязка
        y = z - z_pred

        # Дальше как обычно
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.x.shape[0])
        self.P = (I - K @ H) @ self.P


# ============================================
# Задача: трекаем объект в 2D
# ============================================

print("=" * 60)
print("EXTENDED KALMAN FILTER")
print("Проблема: объект движется в плоскости (x, y)")
print("но датчик выдает (расстояние, угол) — полярные координаты")
print("=" * 60)
print()


# Состояние: [x, y, vx, vy] — где объект и как быстро движется
# Измерение: [r, θ] — расстояние и угол (радар, например)

def f_transition(x, u=None):
    """Динамика: просто равномерное движение"""
    dt = 1.0
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    return F @ x


def h_measurement(x):
    """Нелинейное измерение: (x,y) → (r,θ)"""
    px, py = x[0, 0], x[1, 0]
    r = np.sqrt(px ** 2 + py ** 2)
    theta = np.arctan2(py, px)
    return np.array([[r], [theta]])


def F_jacobian_func(x):
    """Якобиан динамики (тут она линейная, так что просто F)"""
    dt = 1.0
    return np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def H_jacobian_func(x):
    """
    Якобиан измерений (производные полярных координат)

    r = sqrt(x² + y²)
    θ = atan2(y, x)

    Производные:
    ∂r/∂x = x/r,  ∂r/∂y = y/r
    ∂θ/∂x = -y/r²,  ∂θ/∂y = x/r²
    """
    px, py = x[0, 0], x[1, 0]
    r = np.sqrt(px ** 2 + py ** 2)

    # Защита от деления на ноль
    if r < 1e-6:
        r = 1e-6

    H = np.array([[px / r, py / r, 0, 0],
                  [-py / (r ** 2), px / (r ** 2), 0, 0]])
    return H


# Настройки
Q = np.eye(4) * 0.1  # небольшой шум в динамике
R = np.array([[0.5, 0],  # расстояние измеряем довольно точно
              [0, 0.1]])  # угол — точнее

x0 = np.array([[0], [0], [1], [1]])  # старт из (0,0), движется по диагонали
P0 = np.eye(4) * 1.0

# Создаем фильтр
ekf = ExtendedKalmanFilter(f_transition, h_measurement,
                           F_jacobian_func, H_jacobian_func,
                           Q, R, x0, P0)

print("Конфигурация EKF:")
print(f"  Отслеживаем: [x, y, vx, vy]")
print(f"  Датчик выдает: [расстояние, угол]")
print(f"  Шум динамики:\n{Q}")
print(f"  Шум датчика:\n{R}")
print()

# ============================================
# Запускаем симуляцию
# ============================================

np.random.seed(42)
n_steps = 50

true_states = []
measurements_polar = []
estimated_states = []

# Реальное начальное состояние
x_true = np.array([[0], [0], [1.0], [1.0]])

print("Симулируем нелинейную систему...")

for step in range(n_steps):
    # Объект движется
    x_true = f_transition(x_true)
    x_true += np.random.multivariate_normal([0, 0, 0, 0], Q).reshape(4, 1)
    true_states.append(x_true.copy())

    # Датчик видит в полярных координатах (с шумом)
    z_true = h_measurement(x_true)
    z_noisy = z_true + np.random.multivariate_normal([0, 0], R).reshape(2, 1)
    measurements_polar.append(z_noisy.copy())

    # EKF обрабатывает
    ekf.predict()
    ekf.update(z_noisy)
    estimated_states.append(ekf.x.copy())

print(f"✓ Прогнали {n_steps} шагов")
print()

# ============================================
# Смотрим результаты
# ============================================

true_states = np.array([s.flatten() for s in true_states])
estimated_states = np.array([s.flatten() for s in estimated_states])

# Евклидова ошибка позиции
position_errors = np.sqrt((true_states[:, 0] - estimated_states[:, 0]) ** 2 +
                          (true_states[:, 1] - estimated_states[:, 1]) ** 2)

print("=" * 60)
print("РЕЗУЛЬТАТЫ EKF")
print("=" * 60)
print(f"Средняя ошибка:    {np.mean(position_errors):.4f} м")
print(f"Макс ошибка:       {np.max(position_errors):.4f} м")
print(f"Финиш (реально):   ({true_states[-1, 0]:.2f}, {true_states[-1, 1]:.2f})")
print(f"Финиш (оценка):    ({estimated_states[-1, 0]:.2f}, {estimated_states[-1, 1]:.2f})")
print("=" * 60)
print()

# ============================================
# Рисуем
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Слева: траектории
ax1 = axes[0]
ax1.plot(true_states[:, 0], true_states[:, 1],
         'k-', label='Реальный путь', linewidth=2.5)
ax1.plot(estimated_states[:, 0], estimated_states[:, 1],
         'g-', label='EKF оценка', linewidth=2.5)

# Переводим измерения обратно в декартовы (чтобы показать на графике)
meas_x = [m[0, 0] * np.cos(m[1, 0]) for m in measurements_polar]
meas_y = [m[0, 0] * np.sin(m[1, 0]) for m in measurements_polar]
ax1.scatter(meas_x, meas_y, c='red', s=20, alpha=0.4, label='Измерения (r,θ)')

ax1.set_xlabel('X, м', fontsize=12)
ax1.set_ylabel('Y, м', fontsize=12)
ax1.set_title('EKF с нелинейными измерениями', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# Справа: ошибка
ax2 = axes[1]
ax2.plot(range(n_steps), position_errors, 'b-', linewidth=2)
ax2.axhline(y=np.mean(position_errors), color='red',
            linestyle='--', label=f'Среднее: {np.mean(position_errors):.3f} м')
ax2.set_xlabel('Шаг', fontsize=12)
ax2.set_ylabel('Ошибка, м', fontsize=12)
ax2.set_title('Точность EKF', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ekf_result.png', dpi=300)
print("✓ График: 'ekf_result.png'")
plt.show()
