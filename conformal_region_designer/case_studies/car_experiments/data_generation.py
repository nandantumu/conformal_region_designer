import random
import numpy as np
from typing import List

from .car_env import CarEnvironment

# Constants / Building Blocks
TURN_VALUE = 0.6
ACCEL_VALUE = 0.5
INITIAL_CONDITIONS = np.array([0.0, 0.0, 0.0, 0.5])
COAST = np.array([0.0, 0.0])
LEFT = np.array([TURN_VALUE, 0.0])
RIGHT = np.array([-TURN_VALUE, 0.0])
ACCELERATE = np.array([0.0, ACCEL_VALUE])
DECELERATE = np.array([0.0, -ACCEL_VALUE])

# Reference Policies
STRAIGHT_ACCEL = [ACCELERATE]*5 + [COAST]*15
STRAIGHT_ACCEL_LEFT = [ACCELERATE]*5 + [ACCELERATE + LEFT]*15
STRAIGHT_ACCEL_RIGHT = [ACCELERATE]*5 + [ACCELERATE + RIGHT]*15

# Randomness
SEED = 42
STD_DEVS = [TURN_VALUE/2.5, ACCEL_VALUE/2.5]


class ActionPolicy:
    def __init__(self, controls: List[np.ndarray], std_dev: List[float]):
        self.controls = controls
        self.std_dev = std_dev

    def generate_noisy_control(self, control: np.ndarray) -> np.ndarray:
        noisy_control = np.array([random.gauss(control[0], self.std_dev[0]), random.gauss(control[1], self.std_dev[1])])
        return noisy_control

    def generate_controls(self) -> List[np.ndarray]:
        controls = [self.generate_noisy_control(control) for control in self.controls]
        return controls


def create_dataset(action_policy: ActionPolicy, initial_conditions: np.ndarray, num_rollouts: int) -> np.ndarray:
    # Create an instance of the CarEnvironment class
    random.seed(SEED)
    env = CarEnvironment(initial_conditions)
    controls = action_policy.generate_controls()

    # Create an empty list to store the state-action pairs
    dataset = np.zeros((num_rollouts, len(action_policy.controls), 4))

    # Execute the controls in the environment for the specified number of rollouts
    for i in range(num_rollouts):
        # Reset the environment
        controls = action_policy.generate_controls()
        env.reset(initial_conditions)

        # Iterate over the controls and execute them in the environment
        for j, control in enumerate(controls):
            action = control
            next_state = env.step(action)

            # Store the state-action pair in the calibration dataset
            dataset[i, j] = next_state
            
    return dataset

def generate_calibration_dataset(num_rollouts: int) -> np.ndarray:
    action_policy = ActionPolicy(STRAIGHT_ACCEL_LEFT, STD_DEVS)
    dataset = create_dataset(action_policy, INITIAL_CONDITIONS, num_rollouts//2)

    action_policy = ActionPolicy(STRAIGHT_ACCEL_RIGHT, STD_DEVS)
    dataset_2 = create_dataset(action_policy, INITIAL_CONDITIONS, num_rollouts//2)

    # Combine the two datasets on the first axis
    dataset = np.concatenate((dataset, dataset_2), axis=0)
    x = dataset[:, :5, :]
    y = dataset[:, 5:, :]
    return (x, y)

def generate_train_dataset(num_rollouts: int) -> np.ndarray:
    action_policy = ActionPolicy(STRAIGHT_ACCEL, STD_DEVS)
    dataset = create_dataset(action_policy, INITIAL_CONDITIONS, num_rollouts)
    x = dataset[:, :5, :]
    y = dataset[:, 5:, :]
    return (x, y)