import numpy as np

class KalmanFilter:
    def __init__(self, dt=1):
        self.dt = dt
        self.x = np.zeros((4, 1))  # [position, velocity, acc, bias]
        self.P = np.eye(4) * 1000
        self.F = np.array([[1, dt, 0.5*dt**2, 0],
                           [0, 1, dt, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0]])
        self.R = np.array([[100]])  # Measurement noise
        self.Q = np.eye(4)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
