import matplotlib.pyplot as plt
import seaborn as sns
from gym_brt.data.config.configuration import FREQUENCY

from visiontostate.data_collector import DataCollector

sns.set()
sns.set(palette="deep", style="darkgrid")

from visiontostate.vision_learner import VisionLearner
from gym_brt.blackfly.image_preprocessor import IMAGE_SHAPE
from visiontostate.state_estimator_runner import StateEstimatorRunner

"""
Functionalities of the different modules are shown here. Parameters need to be set consistently for all three 
functions to work properly.
"""


def collect_data():
    ######################################################################
    # Collect a dataset for learning a state estimator
    data_id = -1  # -1 to create a new dataset, otherwise add to specified dataset.
    frequency = 200  # works best with 120, otherwise amplitude of SignalGenerator1 might has to be adjusted
    duration = 1680  # duration in sec
    up_bias = 0.5  # how much of the recording should be for small alpha angles
    ######################################################################
    with DataCollector(data_id, frequency) as collector:
        collector.run(duration, up_bias)


def learn():
    ######################################################################
    # Learn a state estimator based on dataset
    dataID = 0  # choose dataset
    shuffle = True  # shuffle dataset

    # HYPERPARAMTERS for training
    epochs = 100
    augmentation = True  # augment image data (parameters can be changed in VisionLearner)
    batch_size = 8
    val_size = 0.05  # random train validation split of the data set

    model_name = None  # to continue training specify a model name in the directory of data set
    ######################################################################

    learner = VisionLearner(dataID, shuffle, epochs, augmentation, batch_size, val_size)
    learner.train(load_model=model_name)


def predict():
    ######################################################################
    # Run a Swingup and Balance controller based on a given trained model
    dataID = 0  # dataset where to load models from
    model_name = 'model000'  # model name in root folder of data_id
    frequency = FREQUENCY  # frequency of pendulum, 100 to 120 Hz is feasible
    duration = 20  # duration in seconds
    ######################################################################

    predictor = StateEstimatorRunner(
        data_id=dataID,
        model_name=model_name,
        image_shape=IMAGE_SHAPE,
        frequency=frequency,
        time_episode=duration)
    predictor.run()


if __name__ == '__main__':
    collect_data()
    plt.show()
