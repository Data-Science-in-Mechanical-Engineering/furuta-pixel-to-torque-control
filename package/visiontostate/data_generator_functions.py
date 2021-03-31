"""
Different signal generators to create a rich dataset for training
generators are described in documentation

@Author: Steffen Bleher
"""
import numpy as np
from gym_brt import configuration, FREQUENCY, PIDCtrl
from gym_brt.control import QubeFlipUpControl
from scipy import signal as s

from tqdm import trange
from gym_brt.envs.reinforcementlearning_extensions.rl_gym_classes import QubeBeginDownEnv

class SwingUpAndHoldGenerator(QubeFlipUpControl):
    # flip up and hold
    # records directly after t_start
    step = 0
    t_start = 0 # sec

    def __init__(self, env=None, action_shape=None, sample_freq=1000, **kwargs):
        super(SwingUpAndHoldGenerator, self).__init__(env=env, sample_freq=sample_freq)
        self.sample_freq = sample_freq
        self.step = self.sample_freq * self.t_start -1

    def action(self, state):
        self.step += 1
        return super().action(state)


class SmallAlphaBiasGenerator(QubeFlipUpControl):
    # does flip up and hold but adds a chirp signal on the input and sweeps over theta
    # records when up
    step = 0  # steps while holding
    t_start = 2  # sec
    theta_amplitude = 30 / 180 * np.pi  # rad
    theta_time_constant = 30  # sec
    f0 = 2.4
    f1 = 2.4

    def __init__(self, env=None, action_shape=None, sample_freq=1000, **kwargs):
        super(SmallAlphaBiasGenerator, self).__init__(env=env, sample_freq=sample_freq)
        self.sample_freq = sample_freq
        self.amplitude = 3.5 * sample_freq/150

    def action(self, state):
        # Get the angles
        theta_x = state[0]
        theta_y = state[1]
        alpha_x = state[2]
        alpha_y = state[3]
        theta = np.arctan2(theta_y, theta_x)
        alpha = np.arctan2(alpha_y, alpha_x)
        theta_dot = state[4]
        alpha_dot = state[5]

        return super().action([theta, alpha, theta_dot, alpha_dot])

    def _action_hold(self, theta, alpha, theta_dot, alpha_dot):
        # theta ref generator

        if self.t_start * self.sample_freq < self.step:
            theta_ref = self.theta_amplitude * np.sin(
                2 * np.pi * self.step / self.sample_freq / self.theta_time_constant)
        else:
            theta_ref = 0

        action = \
            (theta - theta_ref) * self.kp_theta + \
            alpha * self.kp_alpha + \
            theta_dot * self.kd_theta + \
            alpha_dot * self.kd_alpha

        # add chrip signal
        if self.t_start * self.sample_freq < self.step:
            chirp = self.amplitude * s.chirp(self.step / self.sample_freq, self.f0, 10000000, self.f1)
            action += chirp
        self.step += 1
        return action


class FullSpaceGenerator(QubeFlipUpControl):
    # does flip up and lets it fall again
    # records right away, creates data for alpha in [-180,180] and theta in approx [-50,50]
    step = 0  # steps while holding
    t_start = 0  # sec
    duration = 100000  # sec
    theta_amplitude = 60 / 180 * np.pi  # rad
    theta_time_constant = 20  # sec
    t_break = 1
    break_step = 0

    def __init__(self, env=None, action_shape=None, sample_freq=1000, **kwargs):
        super(FullSpaceGenerator, self).__init__(env=env, sample_freq=sample_freq)
        self.sample_freq = sample_freq
        self.step = self.sample_freq * self.t_start-1
        self.break_steps = self.t_break * self.sample_freq

    def offset_pid(self, x, theta_offset):
        pid_controller = PIDCtrl(self.sample_freq, [0.5, 0.0, 0.05, 0.0], th_des=theta_offset)
        return pid_controller(x)[0]

    def action(self, state):
        # Get the angles
        theta_x = state[0]
        theta_y = state[1]
        alpha_x = state[2]
        alpha_y = state[3]
        theta = np.arctan2(theta_y, theta_x)
        alpha = np.arctan2(alpha_y, alpha_x)
        theta_dot = state[4]
        alpha_dot = state[5]

        # If pendulum is within 20 degrees of upright, enable balance control
        if np.abs(alpha) <= (5.0 * np.pi / 180.0):
            self.break_step += 1
            action = 0 #self._action_hold(theta, alpha, theta_dot, alpha_dot)
        else:
            if 0 < self.break_step < self.break_steps:
                if self.break_step == self.break_steps-1:
                    self.break_steps = int(self.sample_freq * (np.random.rand(1)*0.5+1))
                action = 0

                self.break_step += 1
            else:
                action = self._flip_up(theta, alpha, theta_dot, alpha_dot)
                self.break_step = 0
        self.step += 1

        theta_ref = self.theta_amplitude * np.sin(2 * np.pi * self.step / self.sample_freq / self.theta_time_constant)

        voltages = np.array([action + self.offset_pid([theta, alpha, theta_dot, alpha_dot], theta_ref)], dtype=np.float64)
        # set the saturation limit to +/- the Qube saturation voltage
        np.clip(voltages, -configuration.QUBE_MAX_VOLTAGE, configuration.QUBE_MAX_VOLTAGE, out=voltages)
        assert voltages.shape == self.action_shape
        return voltages

    def predict(self, state):
        return self.action(state)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    num_steps = FREQUENCY * 15
    with QubeBeginDownEnv(use_simulator=False, frequency=FREQUENCY) as env:
        alphalist = []

        ctrl_sys = SmallAlphaBiasGenerator(env, sample_freq=FREQUENCY)
        env.reset()
        state, reward, done, info = env.step(np.array([0], dtype=np.float64))
        for step in trange(num_steps):
            # apply signal
            action = ctrl_sys.action(state)
            # get feedback
            state, reward, done, info = env.step(action)
            alphalist.append(env._theta)

        plt.plot(alphalist)
        plt.show()
