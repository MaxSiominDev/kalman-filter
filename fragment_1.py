import numpy as np


class KalmanFilter1D:
    """
    Фильтр Калмана для линейных систем

    Динамика: x_k = F*x_{k-1} + B*u_k + w_k
    Измерения: z_k = H*x_k + v_k
    Шумы: w_k ~ N(0, Q), v_k ~ N(0, R)
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

        self.n = F.shape[0]  # размерность состояния
        self.m = H.shape[0]  # размерность измерений

        # проверка размерностей
        assert F.shape == (self.n, self.n), "F должна быть квадратной n×n"
        assert H.shape[1] == self.n, "У H должно быть n столбцов"
        assert Q.shape == (self.n, self.n), "Q должна быть квадратной n×n"
        assert R.shape == (self.m, self.m), "R должна быть квадратной m×m"
        assert x0.shape == (self.n, 1), "x0 должен быть вектором-столбцом n×1"
        assert P0.shape == (self.n, self.n), "P0 должна быть квадратной n×n"

    def predict(self, u=None, B=None):
        """
        Шаг предсказания
        Прогнозируем состояние: x̂ = F * x + B * u
        Обновляем ковариацию: P = F * P * F^T + Q
        """
        if u is not None and B is not None:
            self.x = self.F @ self.x + B @ u
        else:
            self.x = self.F @ self.x

        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        Шаг коррекции
        Корректируем предсказание по измерениям
        """
        assert z.shape == (self.m, 1), f"z должен быть {self.m}×1"

        # невязка
        y = z - self.H @ self.x

        # ковариация невязки
        S = self.H @ self.P @ self.H.T + self.R

        # усиление Калмана
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # обновляем состояние
        self.x = self.x + K @ y

        # обновляем ковариацию
        I = np.eye(self.n)
        self.P = (I - K @ self.H) @ self.P

        return y, S

    def step(self, z, u=None, B=None):
        """
        Полный цикл фильтрации (предсказание + коррекция)
        """
        self.predict(u, B)
        self.update(z)
        return self.x.copy()

    def get_state(self):
        return self.x.copy()

    def get_covariance(self):
        return self.P.copy()


if __name__ == "__main__":
    print("Запуск теста фильтра Калмана\n")

    # тестовый пример: отслеживаем положение и скорость
    dt = 1.0
    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1, 0]])  # измеряем только положение
    Q = np.eye(2) * 0.01
    R = np.array([[0.1]])
    x0 = np.array([[0], [0]])
    P0 = np.eye(2)

    kf = KalmanFilter1D(F, H, Q, R, x0, P0)

    # обрабатываем измерение
    z = np.array([[1.2]])
    kf.predict()
    kf.update(z)

    state = kf.get_state()
    print(f"Состояние после обновления: [{state[0, 0]:.4f}, {state[1, 0]:.4f}]")
    print("\nТест пройден")