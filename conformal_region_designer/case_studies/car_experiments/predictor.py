# import numpy as np
import jax
import jax.numpy as np
import numpy as tnp

class TrajectoryPredictor:
    def __init__(self, dt, predict_steps: int=50):
        self.dt = dt
        self.steps = predict_steps
    
    def _predict_trajectory(self, states: np.ndarray) -> np.ndarray:
        # Calculate the average turn rate and final velocity over the states
        v = states[-1,-1]
        dtheta = (states[-1,2] - states[0,2]) / (states.shape[0] - 1)
        x,y,theta = states[-1,:3]
        
        trajectory = []
        for _ in range(self.steps):
            # Calculate the next state using the constant turn rate and velocity method
            new_x = x + v * np.cos(theta) * self.dt
            new_y = y + v * np.sin(theta) * self.dt
            new_theta = theta + dtheta
            new_velocity = v

            # Update the current state with the new state
            x = new_x
            y = new_y
            theta = new_theta
            v = new_velocity

            # Append the new state to the trajectory
            trajectory.append((new_x, new_y, new_theta, new_velocity))
        
        trajectory = np.array(trajectory)
        return trajectory
    
    def predict_trajectory(self, state: tnp.ndarray) -> tnp.ndarray:
        """Wrapper for jax internal function"""
        return tnp.array(self._predict_trajectory(state))

    def predict_batched_trajectories(self, states: np.ndarray) -> np.ndarray:
        # Use jax vmap to vectorize the predict_trajectory function
        predict_trajectory = jax.vmap(self._predict_trajectory, in_axes=(0,))
        trajectories = predict_trajectory(states)
        return tnp.array(trajectories)
