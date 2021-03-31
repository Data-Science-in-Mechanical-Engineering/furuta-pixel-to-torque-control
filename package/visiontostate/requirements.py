"""
Find Precision requirements for the State Estimator based on noisy encoder values.

@Author: Steffen Bleher
"""

import math
import numpy as np
from gym_brt import QubeFlipUpControl
from gym_brt.envs import QubeBeginDownEnv, QubeBeginUpEnv


class RequirementsEnv(QubeBeginDownEnv):
    def _get_state(self):
        delta_t = 1 / self.frequency
        sigma = 0 / 180 * np.pi

        alpha_noise = np.random.normal(0.0, sigma)
        theta_noise = np.random.normal(0.0, sigma)

        alpha_dot_noise = 0.5 * alpha_noise / delta_t  # approximate finite difference noise
        theta_dot_noise = 0.5 * theta_noise / delta_t

        return np.array(
            [self._theta + theta_noise,
             self._alpha + alpha_noise,
             self._theta_dot + theta_dot_noise,
             self._alpha_dot + alpha_dot_noise],
            dtype=np.float64,
        )

class RequirementsEnvExtended(QubeBeginDownEnv):
    def _get_state(self):
        delta_t = 1 / self.frequency
        sigma = 2 / 180 * np.pi

        alpha_noise = np.random.normal(0.0, sigma)
        theta_noise = np.random.normal(0.0, sigma)

        alpha_dot_noise = 0.5 * alpha_noise / delta_t  # approximate finite difference noise
        theta_dot_noise = 0.5 * theta_noise / delta_t

        return np.array(
            [np.cos(self._theta + theta_noise),
             np.sin(self._theta + theta_noise),
             np.cos(self._alpha + alpha_noise),
             np.sin(self._alpha + alpha_noise),
             self._theta_dot + theta_dot_noise,
             self._alpha_dot + alpha_dot_noise],
            dtype=np.float64,
        )

if __name__ == '__main__':
    frequency = 100
    n_trials = 1
    steps = 2048

    with RequirementsEnv(frequency=frequency, use_simulator=False) as env:
        controller = QubeFlipUpControl(sample_freq=frequency)
        for episode in range(n_trials):
            state = env.reset()
            for step in range(steps):
                action = controller.action(state)
                state, reward, done, info = env.step(action)
