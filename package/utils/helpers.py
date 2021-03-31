import os

import pandas as pd
from gym_brt.data.config.configuration import FREQUENCY
from matplotlib import pyplot as plt


def set_new_model_id(path):
    model_id = 0
    for (_, dirs, files) in os.walk(path):
        for dir in dirs:
            try:
                if int(dir[:3]) >= model_id:
                    model_id = int(dir[:3]) + 1
            except:
                continue
    path = os.path.join(path, str(model_id).zfill(3))
    os.mkdir(path)
    return model_id


def num_epochs(path, epoch_length=None, frequency=FREQUENCY):
    number_of_epochs = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if ".zip" in file:
                number_of_epochs += 1
    print("Number of epochs: %d" % number_of_epochs)
    if epoch_length is not None:
        steps = number_of_epochs * epoch_length
        print("Steps: %d" % steps)
        if frequency is not None:
            time = steps / frequency / 60
            print("Time (min): %.2f" % time)


def visualize_progress(path):
    columns = ['approxkl', 'clipfrac', 'ep_len_mean', 'ep_reward_mean',
               'explained_variance', 'fps', 'n_updates', 'policy_entropy',
               'policy_loss', 'serial_timesteps', 'time_elapsed', 'total_timesteps',
               'value_loss']
    # try:
    result_log = pd.read_csv(path + "/result_log.csv")
    fig = plt.figure(figsize=(30, 10))
    for i, column in enumerate(columns):
        ax = fig.add_subplot(3, 5, i + 1)
        ax.plot(result_log[column])
        ax.set_title(column)
    plt.show()


def save_progress(path):
    progress_file = path + "/progress.csv"
    columns = ['approxkl', 'clipfrac', 'ep_len_mean', 'ep_reward_mean',
               'explained_variance', 'fps', 'n_updates', 'policy_entropy',
               'policy_loss', 'serial_timesteps', 'time_elapsed', 'total_timesteps',
               'value_loss']
    if os.path.exists(progress_file):
        try:
            progress = pd.read_csv(progress_file)
            if os.path.exists(path + "/result_log.csv"):
                result_log = pd.read_csv(path + "/result_log.csv")
            else:
                result_log = pd.DataFrame(columns=columns)

            progress = progress.reindex(sorted(progress.columns), axis=1)
            result_log = pd.concat([result_log, progress])
            pd.DataFrame.fillna(result_log, value=0, inplace=True)
            result_log.to_csv(path + "/result_log.csv", index=False)
            print("result data saved ...")
        except:
            pass
        os.remove(progress_file)
