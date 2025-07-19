import numpy as np

class KalmanFilter:
    def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=100.0):
        """
        Kalman Filter for 1D motion tracking (either range or velocity).
        State vector: [position, velocity, acceleration, bias]
        """
        self.dt = dt
        self.x = np.zeros((4, 1))  # Initial state: [pos, vel, acc, bias]
        self.P = np.eye(4) * 1000  # Initial uncertainty

        # State transition matrix F
        self.F = np.array([
            [1, dt, 0.5 * dt**2, 0],
            [0,  1,       dt,    0],
            [0,  0,        1,    0],
            [0,  0,        0,    1]
        ])

        # Measurement matrix H (we observe only position)
        self.H = np.array([[1, 0, 0, 0]])

        # Measurement noise covariance R
        self.R = np.array([[measurement_noise]])

        # Process noise covariance Q
        q = process_noise
        self.Q = q * np.array([
            [dt**4/4, dt**3/2, dt**2/2, 0],
            [dt**3/2, dt**2,   dt,     0],
            [dt**2/2, dt,      1,      0],
            [0,      0,        0,    0.1]  # Low bias noise
        ])

    def predict(self):
        """
        Predict the next state and covariance.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z):
        """
        Update state with new measurement z.
        z: observed position (e.g., from CFAR detection)
        """
        z = np.array([[z]])  # Ensure 2D shape
        y = z - self.H @ self.x                  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x.copy()

