"""
PPO Learner class can train a PPO agent and run predictions on Qube

@Author: Steffen Bleher
"""

#!/usr/bin/env python3
import glob
import time

import numpy as np
import os
from torch import nn

from gym_brt.envs import (
    QubeSwingupEnv,
    QubeBalanceEnv,
    QubeBeginDownEnv,
    QubeBeginUpEnv,
    RandomStartEnv,
    NoisyEnv
)

from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo import MlpPolicy, PPO
from stable_baselines3.common import logger

from gym_brt.envs.reinforcementlearning_extensions.vision_wrapping_classes import VisionQubeBeginDownEnv

from statetoinput.state_estimator_wrapper import VtSQubeBeginDownEnv
from utils.callbacks import CheckpointCallback
from utils.helpers import set_new_model_id, num_epochs, save_progress
from visiontostate.requirements import RequirementsEnv, RequirementsEnvExtended

NET_ARCH = [64, dict(vf=[64, 12], pi=[64, 12])]


class PPOLearner:

    def __init__(self, model_id, env, use_simulator, frequency, n_steps, batch_size, simulation_mode='ode',
                 vision_model_data_id=-1, vision_model_model_name='', n_envs=1, wrapper_cls=None, model_path_ext=None):
        my_path = os.path.abspath(os.path.dirname(__file__))
        self.model_path = os.path.join(my_path, "../../data/statetoinput/")
        if model_path_ext is not None:
            self.model_path = os.path.join(self.model_path, model_path_ext)
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path, exist_ok=True)
        self.use_simulator = use_simulator
        self.model_id = model_id
        self.frequency = frequency
        self._n_steps = n_steps
        self._batch_size = batch_size
        self.model = None
        tag = env + ":"
        if use_simulator:
            tag += "sim:"
        else:
            tag += "real:"
        tag += "freq" + str(frequency) + ':'
        self.tag = tag
        if self.model_id is not None:
            if self.model_id == -1:
                print("Start learning a new model with task: " + env)
                self.model_id = set_new_model_id(self.model_path)
                self.new = True
            else:
                print("Start learning on model with id " + str(model_id) + " and task: " + env)
                self.new = False

            self.model_path += str(self.model_id).zfill(3)

            open(self.model_path + "/tag_" + self.tag, "w+")
        else:
            self.new = False

        envs = {
            "QubeSwingupEnv": QubeSwingupEnv,
            "QubeBalanceEnv": QubeBalanceEnv,
            "QubeBeginDownEnv": QubeBeginDownEnv,
            "QubeBeginUpEnv": QubeBeginUpEnv,
            "RandomStartEnv": RandomStartEnv,
            "NoisyEnv": NoisyEnv,
            "VtSQubeBeginDownEnv": VtSQubeBeginDownEnv,
            "VisionQubeBeginDownEnv": VisionQubeBeginDownEnv,
            "RequirementsEnv": RequirementsEnv,
            "RequirementsEnvExtended": RequirementsEnvExtended
        }

        output_formats = ["log", "csv", "stdout"]
        logger.configure(self.model_path, output_formats)

        if env is 'VtSQubeBeginDownEnv':
            env = envs[env]
            self.env = env(vision_model_data_id, vision_model_model_name, use_simulator=use_simulator, simulation_mode=simulation_mode,
                           batch_size=self._n_steps)
        else:
            env = envs[env]
            env_kwargs = {"use_simulator": use_simulator,
                          "simulation_mode": simulation_mode,
                          "batch_size": self._n_steps,
                          'frequency': self.frequency,
                          "integration_steps": 1
                          }

            if use_simulator and n_envs > 1:
                from stable_baselines3.common.env_util import make_vec_env

                self.env = make_vec_env(env, n_envs, env_kwargs=env_kwargs,
                                        wrapper_class=wrapper_cls, vec_env_cls=SubprocVecEnv)
            elif use_simulator and n_envs == 1:
                self.env = env(**env_kwargs)
                self.env = wrapper_cls(self.env)
            else:
                # self.env = CalibrationWrapper(env(**env_kwargs), noise=False)
                self.env = env(**env_kwargs)
                self.env = Monitor(self.env, self.model_path + "/logs")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        try:
            self.env.unwrapped.__exit__(None, None, None)
        except AttributeError:
            pass
        path = os.path.join(self.model_path, str(self.model_id).zfill(3))
        num_epochs(path, self._n_steps, self.frequency)
        save_progress(self.model_path)

    def train(self, num_timesteps, save_interval, act_fun=nn.Tanh, net_arch=NET_ARCH, force_load_from=None):
        # Round save interval to a multiple of 2048
        save_interval = int(np.ceil(save_interval / 2048))

        tb_logdir = self.model_path + "/tb"

        if not self.new:
            if self.model_id == None:
                my_path = os.path.abspath(os.path.dirname(__file__))
                load_path = os.path.join(my_path, "working_models/model.zip")
                print("Loaded PPO from working models ... ")
            else:
                load_path = self.model_path
                n = 0
                for x, _, _ in os.walk(load_path):
                    if "checkpoints0" in x:
                        if len(x) > 11 and int(x[-2:]) > n:
                            n = int(x[-2:])
                if len(os.listdir(load_path + "/checkpoints/")) == 0:
                    load_path = os.path.join(load_path, "checkpoints0" + str(n).zfill(2))
                elif num_timesteps == 0:
                    # if running use checkpoint files
                    load_path = self.model_path + "/checkpoints/"
                else:
                    n += 1
                    old_load_path = os.path.join(load_path, "checkpoints0" + str(n).zfill(2))

                    os.rename(load_path + "/checkpoints", old_load_path)
                    os.mkdir(load_path + "/checkpoints")
                    load_path = old_load_path

                list_of_files = glob.glob(load_path + "/*")
                print(load_path)
                load_path = max(list_of_files, key=os.path.getctime)
                print("Loading model from: " + load_path)
        else:
            load_path = None

        if force_load_from is not None:
            load_path = force_load_from

        policy = MlpPolicy #CustomActorCriticPolicy
        model = PPO(
            policy=policy,
            env=self.env,
            n_steps=self._n_steps,
            batch_size=self._batch_size,
            gae_lambda=0.98,
            gamma=0.995,
            n_epochs=10,
            ent_coef=0.0, #0.01
            learning_rate=2e-4,  # 3e-4, 2.5e-4
            clip_range=0.1,
            verbose=1,
            tensorboard_log=tb_logdir,
            policy_kwargs=dict(activation_fn=act_fun, net_arch=net_arch)
        )

        if save_interval > 0:
            callback = CheckpointCallback(save_freq=self._n_steps, save_path=self.model_path + '/checkpoints/')
            # callback = init_save_callback(self.model_path, 2048, save_interval)
        else:
            callback = None

        # Optionally load before or save after training
        if load_path is not None:
            model = model.load(load_path, env=self.env)

        if num_timesteps > 0:
            model.learn(total_timesteps=num_timesteps, callback=callback)

        model.save(self.model_path + "/model")

        self.model = model
        return load_path

    def action(self, obs):
        if self.model is None:
            self.train(0, False)
            self.step = 0
        action, _ = self.model.predict(obs)
        self.step += 1
        return action

    def predict(self, num_timesteps, net_arch=NET_ARCH, act_fun=nn.Tanh, render=False):
        # Load model
        self.train(0, False, net_arch=net_arch, act_fun=act_fun)

        time_list = []
        alpha_list = []
        alpha_predict_list = []
        theta_list = []
        theta_predict_list = []
        alpha_vel_list = []
        theta_vel_list = []
        alpha_vel_predict_list = []
        theta_vel_predict_list = []

        print('Start ...')

        t = time.time()

        obs = np.zeros(self.env.observation_space.shape)
        obs[:] = self.env.reset()
        ret = 0
        for _ in range(num_timesteps):
            action, _ = self.model.predict(obs)
            state_predict, reward, done, info = self.env.step(action)
            obs[:] = state_predict
            ret += reward
            if done:
                break
            if self.use_simulator and render:
                self.env.render()

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

        # for _ in range(self.frequency):
        #     self.env.step(np.array([0]))

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
