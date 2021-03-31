"""
Runs a real-time experiment based on a full_state model. Model gets loaded and with run() a experiment can be done
and it will be visualized.

@Author: Steffen Bleher
"""
import time

import numpy as np
from gym_brt.envs import convert_state

from gym_brt.envs.reinforcementlearning_extensions.vision_wrapping_classes import VisionQubeBeginDownEnv
from visiontostate.data_generator_functions import SwingUpAndHoldGenerator
from visiontostate.vision_predictor import VisionPredictor


class StateEstimatorRunner:

    def __init__(self, data_id, model_name, time_episode, image_shape, frequency):
        self.data_id = data_id
        self.time_episode = time_episode
        self.frequency = frequency
        self.image_shape = image_shape
        self.model_name = model_name

    def run(self):
        num_steps = self.frequency * self.time_episode
        time_list = []
        alpha_list = []
        alpha_predict_list = []
        theta_list = []
        theta_predict_list = []
        alpha_vel_list = []
        theta_vel_list = []
        alpha_vel_predict_list = []
        theta_vel_predict_list = []
        state_predict = np.zeros(10)
        predicting = False

        with VisionQubeBeginDownEnv(use_simulator=False, frequency=self.frequency) as env, \
                VisionPredictor(self.data_id, self.model_name, self.image_shape, self.frequency, predict_state=True) as pred:

            print('Start ...')

            ctrl_sys = SwingUpAndHoldGenerator(env, sample_freq=self.frequency)

            t = time.time()
            env.reset()
            state, reward, done, info = env.step(np.array([0], dtype=np.float64))
            encoder_state = [info['theta'], info['alpha'], info['theta_dot'], info['alpha_dot']]

            for step in range(num_steps):
                # apply signal
                if predicting:
                    conv_state = convert_state(state_predict)
                    action = ctrl_sys.action(conv_state)
                else:
                    print('not predicting')
                    action = ctrl_sys.action(encoder_state)
                # get feedback
                state, reward, done, info = env.step(action)
                encoder_state = [info['theta'], info['alpha'], info['theta_dot'], info['alpha_dot']]

                # store data
                if ctrl_sys.step == ctrl_sys.sample_freq * ctrl_sys.t_start:
                    print("Predicting ...")
                    alpha_predict_list = []
                    theta_predict_list = []
                    alpha_list = []
                    theta_list = []
                    time_list = []
                    alpha_vel_list = []
                    theta_vel_list = []
                    alpha_vel_predict_list = []
                    theta_vel_predict_list = []
                if ctrl_sys.step >= ctrl_sys.sample_freq * ctrl_sys.t_start:
                    predicting = True

                    # get image, preprocess and predict
                    state_predict = pred.predict(state)

                alpha = info['alpha']
                theta = info['theta']
                alpha_predict = np.arctan2(state_predict[3], state_predict[2])
                theta_predict = np.arctan2(state_predict[1], state_predict[0])

                time_list.append(time.time() - t)
                alpha_list.append(np.rad2deg(alpha))
                theta_list.append(np.rad2deg(theta))
                alpha_predict_list.append(np.rad2deg(alpha_predict))
                theta_predict_list.append(np.rad2deg(theta_predict))
                theta_vel_list.append(info['theta_dot'])
                alpha_vel_list.append(info['alpha_dot'])
                theta_vel_predict_list.append(state_predict[4])
                alpha_vel_predict_list.append(state_predict[5])

            print("Time needed: %5.2f" % (time.time() - t))
            print("Time expected: %5.2f" % (self.time_episode))

            for _ in range(self.frequency):
                env.step([0])

        alpha_predict_list.pop(0)
        theta_predict_list.pop(0)
        alpha_list.pop(0)
        theta_list.pop(0)
        time_list.pop(0)
        theta_vel_list.pop(0)
        alpha_vel_list.pop(0)
        theta_vel_predict_list.pop(0)
        alpha_vel_predict_list.pop(0)

        np.save("plotting/dump/alpha_predict_list.npy", np.asarray(alpha_predict_list))
        np.save("plotting/dump/alpha_list.npy", np.asarray(alpha_list))
        np.save("plotting/dump/alpha_vel_predict_list.npy", np.asarray(alpha_vel_predict_list))
        np.save("plotting/dump/alpha_vel_list.npy", np.asarray(alpha_vel_list))