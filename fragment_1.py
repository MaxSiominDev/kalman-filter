import numpy as np


class KalmanFilter1D:
    """
    Фильтр Калмана для линейных систем

    Уравнения системы:
    x_k = F*x_{k-1} + B*u_k + w_k  (динамика)
    z_k = H*x_k + v_k              (измерения)
    w_k ~ N(0, Q), v_k ~ N(0, R)   (шумы)
    """

    def __init__(self, F, H, Q, R, x0, P0):
        """
        F  - матрица перехода (n×n)
        H  - матрица измерений (m×n)
        Q  - ковариация шума процесса (n×n)
        R  - ковариация шума измерений (m×m)
        x0 - начальное состояние (n×1)
        P0 - начальная ковариация (n×n)
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

        self.n = F.shape[0]  # размер вектора состояния
        self.m = H.shape[0]  # размер вектора измерений

        # проверяем, что размерности сходятся
        assert F.shape == (self.n, self.n), "F должна быть квадратной n×n"
        assert H.shape[1] == self.n, "У H должно быть n столбцов"
        assert Q.shape == (self.n, self.n), "Q должна быть квадратной n×n"
        assert R.shape == (self.m, self.m), "R должна быть квадратной m×m"
        assert x0.shape == (self.n, 1), "x0 должен быть вектором-столбцом n×1"
        assert P0.shape == (self.n, self.n), "P0 должна быть квадратной n×n"

    def predict(self, u=None, B=None):
        """
        Шаг предсказания

        Прогнозируем состояние на следующий момент времени:
        x̂ = F * x + B * u
        P = F * P * F^T + Q
        """
        # считаем новое состояние
        if u is not None and B is not None:
            self.x = self.F @ self.x + B @ u
        else:
            self.x = self.F @ self.x

        # обновляем ковариацию
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        Шаг коррекции по измерению

        Корректируем предсказание с учетом измерений:
        y = z - H*x           (невязка)
        S = H*P*H^T + R       (ковариация невязки)
        K = P*H^T*S^(-1)      (коэффициент усиления)
        x = x + K*y           (обновленное состояние)
        P = (I - K*H)*P       (обновленная ковариация)
        """
        assert z.shape == (self.m, 1), f"z должен быть {self.m}×1"

        # невязка между измерением и предсказанием
        y = z - self.H @ self.x

        # ковариация невязки
        S = self.H @ self.P @ self.H.T + self.R

        # коэффициент усиления Калмана
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # корректируем состояние
        self.x = self.x + K @ y

        # обновляем ковариацию
        I = np.eye(self.n)
        self.P = (I - K @ self.H) @ self.P

        return y, S  # можно использовать для диагностики

    def step(self, z, u=None, B=None):
        """
        Полный цикл: предсказание → коррекция

        Возвращает обновленную оценку состояния
        """
        self.predict(u, B)
        self.update(z)
        return self.x.copy()

    def get_state(self):
        """Текущая оценка состояния"""
        return self.x.copy()

    def get_covariance(self):
        """Текущая матрица ковариации"""
        return self.P.copy()


if __name__ == "__main__":
    print("Тест KalmanFilter1D...")

    # Пример: отслеживаем положение и скорость объекта
    dt = 1.0
    F = np.array([[1, dt], [0, 1]])  # положение зависит от скорости
    H = np.array([[1, 0]])  # измеряем только положение
    Q = np.eye(2) * 0.01  # небольшой шум процесса
    R = np.array([[0.1]])  # шум измерений
    x0 = np.array([[0], [0]])  # начинаем с нуля
    P0 = np.eye(2)  # начальная неопределенность

    kf = KalmanFilter1D(F, H, Q, R, x0, P0)

    # пробуем обработать измерение
    z = np.array([[1.2]])
    kf.predict()
    kf.update(z)

    print(f"Состояние после обновления: {kf.get_state().T}")
    print("Всё работает")
