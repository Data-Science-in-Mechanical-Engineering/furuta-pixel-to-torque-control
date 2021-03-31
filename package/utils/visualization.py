import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import typing as tp
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def plot_evaluation_mean(path: tp.AnyStr, include: tp.AnyStr = None, save_path: tp.AnyStr = None, timesteps: bool = True):

    for exp_folder in sorted(os.listdir(path)):
        if exp_folder.startswith('_'):
            continue

        if not exp_folder.startswith('000-normal-simulation'):
            if include is not None and include not in exp_folder:
                continue

        y = []
        subpath = f"{path}/{exp_folder}/"
        for model_folder in sorted(os.listdir(subpath), key=lambda i: int(i[:3])):
            if not model_folder.startswith('0'):
                continue
            df = pd.read_csv(subpath + model_folder + '/run-PPO_1-tag-rollout_ep_rew_mean.csv')
            x = df['Step'] if timesteps else df['Step'] // df['Step'][0]
            y.append(df['Value'])

        y = np.asarray(y)
        y_max = np.max(y, axis=0)
        y_min = np.min(y, axis=0)
        y_mean = np.mean(y, axis=0)

        # plot
        plt.plot(x, y_mean, label=exp_folder[4:])
        plt.fill_between(x, y_min, y_max, alpha=0.2)

    plt.legend(loc="upper left")
    if timesteps:
        plt.xlabel('timestep')
    else:
        plt.xlabel('iteration')
    plt.ylabel('return')

    if save_path is not None:
        plt.title(save_path.split('/')[-1])
        plt.savefig(fname=save_path)

    plt.show()
    plt.close()


def plot_evaluation_mean_categorized(path: tp.AnyStr,
                                     categories: tp.List,
                                     include: tp.AnyStr = None,
                                     save_path: tp.AnyStr = None,
                                     timesteps: bool = True
                                     ):

    collection = dict()
    categories.append('normal-simulation')
    for category in categories:
        collection.update({category: []})

    for exp_folder in sorted(os.listdir(path)):
        if exp_folder.startswith('_'):
            continue

        if not exp_folder.startswith('000-normal-simulaton'):
            if include is not None and include not in exp_folder:
                continue

        subpath = f"{path}/{exp_folder}/"
        for model_folder in sorted(os.listdir(subpath), key=lambda i: int(i[:3])):
            if not model_folder.startswith('0'):
                continue
            df = pd.read_csv(subpath + model_folder + '/run-PPO_1-tag-rollout_ep_rew_mean.csv')
            x = df['Step'] if timesteps else df['Step'] // df['Step'][0]

            for category in categories:
                if category in exp_folder:
                    collection[category].append(df['Value'])

    for category in categories:
        y = np.asarray(collection[category])
        y_max = np.max(y, axis=0)
        y_min = np.min(y, axis=0)
        y_mean = np.mean(y, axis=0)

        # plot
        plt.plot(x, y_mean, label=category)
        plt.fill_between(x, y_min, y_max, alpha=0.2)

    plt.legend(loc="upper left")
    if timesteps:
        plt.xlabel('timestep')
    else:
        plt.xlabel('iteration')

    plt.ylabel('return')

    if save_path is not None:
        plt.title(save_path.split('/')[-1])
        plt.savefig(fname=save_path)

    plt.show()
    plt.close()


def plot_evaluation_best(path: tp.AnyStr, include: tp.AnyStr = None, save_path: tp.AnyStr = None, timesteps: bool = True):

    for exp_folder in sorted(os.listdir(path)):
        if exp_folder.startswith('_'):
            continue

        if not exp_folder.startswith('000-normal-simulation'):
            if include is not None and include not in exp_folder:
                continue

        y = []
        subpath = f"{path}/{exp_folder}/"
        for model_folder in sorted(os.listdir(subpath), key=lambda i: int(i[:3])):
            if not model_folder.startswith('0'):
                continue
            df = pd.read_csv(subpath + model_folder + '/run-PPO_1-tag-rollout_ep_rew_mean.csv')
            x = df['Step'] if timesteps else df['Step'] // df['Step'][0]
            y.append(df['Value'])

        y = np.asarray(y)
        y_max = np.max(y, axis=0)
        y_min = np.min(y, axis=0)
        y_best_arg = np.argmax(y[:, -1])

        # plot
        plt.plot(x, y[y_best_arg], label=exp_folder[4:])
        plt.fill_between(x, y_min, y_max, alpha=0.2)

    plt.legend(loc="upper left")
    if timesteps:
        plt.xlabel('timestep')
    else:
        plt.xlabel('iteration')
    plt.ylabel('return')

    plt.show()
    plt.close()


def plot_evaluation_bar(path: tp.AnyStr,
                        include: tp.AnyStr = None,
                        save_path: tp.AnyStr = None,
                        eval_iterations: int = 4
                        ):
    from gym_brt.data.config.configuration import FREQUENCY
    from statetoinput.ppo_learner import PPOLearner
    from gym_brt.envs.reinforcementlearning_extensions import ExponentialRewardWrapper, TrigonometricObservationWrapper

    def wrapper(env):
        env = ExponentialRewardWrapper(env)
        env = TrigonometricObservationWrapper(env)
        return env

    def predict(model_id, model_path_ext=None):
        ############################################
        # QubeBalanceEnv or QubeSwingupEnv for normal states
        # QubeBeginUpEnv or QubeBeginDownEnv for xy angles
        env = "QubeSwingupEnv"
        n_steps = 2048
        batch_size = 32
        use_simulator = True
        simulation_mode = 'mujoco'

        num_evaluationsteps = 2048
        net_arch = [64, 64]
        ############################################

        with PPOLearner(model_id, env, use_simulator, FREQUENCY, n_steps, batch_size, simulation_mode=simulation_mode,
                        n_envs=1, wrapper_cls=wrapper, model_path_ext=model_path_ext) as learner:
            ret = learner.run(num_evaluationsteps, net_arch=net_arch, render=False)
        return ret

    objects, values_mean, values_best = [], [], []
    for exp_folder in sorted(os.listdir(path)):
        if exp_folder.startswith('_'):
            continue

        if not exp_folder.startswith('000-normal-simulation'):
            if include is not None and include not in exp_folder:
                continue

        y, train_return = [], []
        subpath = f"{path}/{exp_folder}/"
        for model_folder in sorted(os.listdir(subpath), key=lambda i: int(i[:3])):
            if not model_folder.startswith('0'):
                continue
            model_id = int(model_folder)
            avg_ret = 0
            for _ in range(eval_iterations):
                avg_ret += predict(model_id, exp_folder + "/")
            avg_ret /= eval_iterations
            y.append(avg_ret)

            df = pd.read_csv(subpath + model_folder + '/run-PPO_1-tag-rollout_ep_rew_mean.csv')
            train_return.append(df['Value'])

        y_mean = np.mean(y)
        y_best = np.max(y)
        objects.append(exp_folder)
        values_mean.append(y_mean)
        values_best.append(y_best)

    positions = np.arange(len(objects))
    # plt.bar(positions, values)
    width = 0.35  # the width of the bars
    plt.bar(positions - width / 2, values_mean, width, label='Mean')
    plt.bar(positions + width / 2, values_best, width, label='Best')
    plt.xticks(positions, objects, rotation='vertical')
    plt.ylabel('return')
    plt.legend(loc="upper right")

    if save_path is not None:
        plt.title(save_path.split('/')[-1])
        plt.savefig(fname=save_path)

    plt.show()
    plt.close()


def plot_importance_bar(path: tp.AnyStr,
                        include: tp.AnyStr = None,
                        save_path: tp.AnyStr = None,
                        eval_iterations: int = 4,
                        best: bool = False
                        ):
    from gym_brt.data.config.configuration import FREQUENCY
    from statetoinput.ppo_learner import PPOLearner
    import torch
    from gym_brt.envs.reinforcementlearning_extensions import ExponentialRewardWrapper, TrigonometricObservationWrapper

    def wrapper(env):
        env = ExponentialRewardWrapper(env)
        env = TrigonometricObservationWrapper(env)
        return env

    def predict(model_id, model_path_ext=None, states=None):
        ############################################
        # QubeBalanceEnv or QubeSwingupEnv for normal states
        # QubeBeginUpEnv or QubeBeginDownEnv for xy angles
        env = "QubeSwingupEnv"
        n_steps = 2048
        batch_size = 32
        use_simulator = True
        simulation_mode = 'mujoco'

        num_timesteps = 2048
        net_arch = [64, 64]
        done = True

        actions = list()
        ############################################

        if states is None:
            states = list()
            done = False

        with PPOLearner(model_id, env, use_simulator, FREQUENCY, n_steps, batch_size, simulation_mode=simulation_mode,
                        n_envs=1, wrapper_cls=wrapper, model_path_ext=model_path_ext) as learner:
            learner.train(0, False, net_arch=net_arch)

            if done:
                obs = states[0]
            else:
                obs = learner.env.reset()
                states.append(obs)
            ret = 0
            for j in range(1, num_timesteps+1):
                action, _ = learner.model.predict(obs)
                actions.append(action)
                if done:
                    obs = states[j]
                else:
                    obs, reward, _, _ = learner.env.step(action)
                    states.append(obs)
                    ret += reward
            _, log_probs, _ = learner.model.policy.evaluate_actions(torch.as_tensor(states[:-1]).to('cuda:0'), torch.as_tensor(actions).to('cuda:0'))
            log_prob_sum = torch.sum(log_probs).item()
        return ret, states, log_prob_sum

    objects, values_mean, values_best = [], [], []
    states_buffer, log_prob_buffer, return_buffer = [], [], []
    for exp_folder in sorted(os.listdir(path)):
        if exp_folder.startswith('_'):
            continue

        if not exp_folder.startswith('000-normal-simulation'):
            if include is not None and include not in exp_folder:
                continue

        y, train_return = [], []
        k = 0
        subpath = f"{path}/{exp_folder}/"
        for model_folder in sorted(os.listdir(subpath), key=lambda i: int(i[:3])):
            if not model_folder.startswith('0'):
                continue
            model_id = int(model_folder)
            importance = 0
            for i in range(eval_iterations):
                if len(states_buffer) < eval_iterations * (model_id + 1):
                    ret, s, log_prob = predict(model_id, exp_folder + "/", None)
                    states_buffer.append(s)
                    log_prob_buffer.append(log_prob)
                    return_buffer.append(ret)
                    importance += ret
                else:
                    _, _, log_prob = predict(model_id, exp_folder + "/", states_buffer[k])
                    importance += np.log(return_buffer[k]) + log_prob - log_prob_buffer[k]
                    k += 1
            importance /= eval_iterations
            y.append(importance)

            df = pd.read_csv(subpath + model_folder + '/run-PPO_1-tag-rollout_ep_rew_mean.csv')
            train_return.append(df['Value'])

        y_mean = np.mean(y)
        y_best = np.max(y)
        objects.append(exp_folder)
        values_mean.append(y_mean)
        values_best.append(y_best)

    positions = np.arange(len(objects))
    #plt.bar(positions, values)
    width = 0.35  # the width of the bars
    plt.bar(positions - width / 2, values_mean, width, label='Mean')
    plt.bar(positions + width / 2, values_best, width, label='Best')
    plt.xticks(positions, objects, rotation='vertical')
    plt.ylabel('return')
    plt.legend(loc="upper right")

    if save_path is not None:
        plt.title(save_path.split('/')[-1])
        plt.savefig(fname=save_path)

    plt.show()
    plt.close()


if __name__ == '__main__':
    path = '../../data/statetoinput'

    timesteps = False
    plot_evaluation_bar(path, include=None, save_path=path+'/_figures/evaluation_bar', eval_iterations=10)
    #plot_evaluation_mean(path, include=None, save_path=path+'/_figures/training_mean_all', timesteps=timesteps)
    #plot_evaluation_mean(path, include='mass', save_path=path+'/_figures/training_mean_mass', timesteps=timesteps)
    #plot_evaluation_mean(path, include='noise', save_path=path+'/_figures/training_mean_noise', timesteps=timesteps)

    #plot_evaluation_best(path, include=None, save_path=path+'/_figures/training_best_all', timesteps=timesteps)
    #plot_evaluation_best(path, include='mass', save_path=path+'/_figures/training_best_mass', timesteps=timesteps)
    #plot_evaluation_best(path, include='noise', save_path=path+'/_figures/training_best_noise', timesteps=timesteps)

    #plot_evaluation_mean_categorized(path, categories=['arm', 'pole', 'motor', 'action', 'observation'], include=None,
    #                                 save_path=path + '/_figures/training_mean_categorized_all', timesteps=timesteps)
