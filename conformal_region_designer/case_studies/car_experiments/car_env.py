import numpy as np

# Constants
L = 1.0  # Wheelbase length
DT = 0.1  # Time step

class CarEnvironment:
    def __init__(self, initial_state):
        self.state = initial_state.copy()
        self.L = L
        self.dt = DT

    def step(self, action):
        # Kinematic vehicle model equations
        x, y, theta, v = self.state
        delta, a = action

        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = v * np.tan(delta) / L
        dv = a

        self.state += self.dt * np.array([dx, dy, dtheta, dv])

        return self.state.copy()

    def reset(self, initial_state):
        self.state = initial_state.copy()


