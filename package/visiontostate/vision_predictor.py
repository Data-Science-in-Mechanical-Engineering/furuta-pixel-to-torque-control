"""
VisionPredictor holds models ready for real-time use and can predict the states (with velocities) from images.
A simple low pass filter is used for the states and the velocities. A more sophisticated filter can be found in the
LowPassFilter class.

@Author: Steffen Bleher
"""
import os
import time

import numpy as np
import torch
from gym_brt.data.config.configuration import FREQUENCY
from matplotlib import pyplot as plt
from scipy import signal

from gym_brt.blackfly.image_preprocessor import IMAGE_SHAPE
from visiontostate.models import VisionToStateNet


class LowPassFilter:
    # real time butterworth low pass filter
    def __init__(self, order, cutoff_frequency, sample_frequency):
        self.b, self.a = signal.butter(N=order, Wn=cutoff_frequency, btype='lowpass', fs=sample_frequency)
        self.x = np.zeros(order + 1)
        self.y = np.zeros(order)

    def filter(self, xn):
        for i in range(self.x.shape[0] - 2, -1, -1):
            self.x[i + 1] = self.x[i]
        self.x[0] = xn
        yn = (np.dot(self.b, self.x) - np.dot(self.a[1:], self.y)) / self.a[0]
        for i in range(self.y.shape[0] - 2, -1, -1):
            self.y[i + 1] = self.y[i]
        self.y[0] = yn
        return yn


class VisionPredictor:

    def __init__(self, data_id, model_name, image_shape=IMAGE_SHAPE, frequency=FREQUENCY, predict_state=True):
        self.data_id = data_id
        self._frequency = frequency
        self._theta = 0
        self._alpha = np.pi  # could cause problems if prediction starts when pendulum up, use = 0
        self._theta_velocity = 0
        self._alpha_velocity = 0
        self._state = self._get_state()
        self.predict_state = predict_state
        self.time_delays = []

        if self.data_id == -1:
            my_path = os.path.abspath(os.path.dirname(__file__))
            self.save_path = os.path.join(my_path, 'working_models/')
        else:
            my_path = os.path.abspath(os.path.dirname(__file__))
            data_path = os.path.join(my_path, '../../data/visiontostate/')
            self.save_path = data_path + str(self.data_id).zfill(3) + '/'

        print("Load model from: " + self.save_path)
        self.model = VisionToStateNet()
        self.model.to(device=torch.device("cuda"))
        self.model.load_state_dict(torch.load(self.save_path + model_name + ".pt"))
        self.model.eval()

        # run first prediction for faster predictions later
        image = np.zeros(shape=image_shape)
        self.image_input = np.zeros([1, image_shape[2], image_shape[0], image_shape[1]])
        self.image_input[0, :, :, :] = image.transpose((2, 0, 1))
        self.model(torch.from_numpy(self.image_input).float().to(torch.device("cuda"))).float()
        self.time_delays = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.time_delays = np.array(self.time_delays)
        mean = np.mean(self.time_delays) * 1000
        std = np.std(self.time_delays) * 1000
        print("Prediction time of VisionToState (ms): mean = " + str(mean) + ", std = " + str(std))
        del self

    def predict_model_output(self, image):
        self.image_input[0, :, :, :] = image.permute(2, 0, 1)
        vec = self.model(torch.from_numpy(self.image_input).float().to(torch.device("cuda"))).float()
        return vec.cpu().detach().numpy()[0]

    def predict(self, image):
        t = time.time()
        vec = self._state[0:4]
        if self.predict_state:
            # TODO optimized prediction
            # run on gpu? torch.cuda.syncronize and extensive warmup to figure out fastest kernel
            self.image_input[0, :, :, :] = image.transpose((2, 0, 1))

            # # Display image
            # print(image.shape)
            # cv2.imshow('frame', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # exit()

            vec = self.model(torch.from_numpy(self.image_input).float().to(torch.device("cuda"))).float()
            vec = vec.cpu().detach().numpy()[0]

        self._update_states(vec)
        self.time_delays.append(time.time() - t)
        return self._get_state()

    def _update_states(self, vec):
        # LOW PASS on cos sin of angle
        c = 0.0
        vec = (1 - c) * vec + c * self._state[0:4]

        alpha = np.arctan2(vec[3], vec[2])
        theta = np.arctan2(vec[1], vec[0])

        # calculate velocities from angles
        # handle jumps from -pi to +pi in alpha to get a continuous alpha_velocity
        if np.abs(alpha - self._alpha) >= np.pi:
            # alpha is crossing +-pi
            dist = min(np.abs(np.pi - alpha), np.abs(-np.pi - alpha)) + min(np.abs(np.pi - self._alpha),
                                                                            np.abs(-np.pi - self._alpha))
            if alpha > self._alpha:
                dist = -dist
            alpha_velocity = dist * self._frequency
        else:
            # alpha is not crossing +-pi
            alpha_velocity = (alpha - self._alpha) * self._frequency

        theta_velocity = (theta - self._theta) * self._frequency

        # LOW PASS on differentiated velocities
        c_alpha = 0.1
        c_theta = 0.2
        self._theta_velocity = (1 - c_theta) * theta_velocity + c * self._theta_velocity
        self._alpha_velocity = (1 - c_alpha) * alpha_velocity + c * self._alpha_velocity

        # upper threshold on velocitites
        if np.abs(self._alpha_velocity) > 50:
            self._alpha_velocity = 0
        if np.abs(self._theta_velocity) > 50:
            self._theta_velocity = 0

        self._theta = theta
        self._alpha = alpha

    def _get_state(self):
        self._state = np.asarray([
            np.cos(self._theta),
            np.sin(self._theta),
            np.cos(self._alpha),
            np.sin(self._alpha),
            self._theta_velocity,
            self._alpha_velocity,
            0,
            0,
            0,
            0,
        ], dtype=np.float32)

        return self._state


if __name__ == '__main__':
    # TEST LOW PASS FILTER
    filt = LowPassFilter(4, 40, 120)

    x = np.load("alpha_predict_list.npy")
    t = np.arange(x.shape[0]) / 120
    x_real = np.load("alpha_list.npy")
    y = np.zeros(x.shape[0])

    for i, _ in enumerate(x):
        y[i] = filt.filter(x[i])

    freq = np.fft.fftfreq(t.shape[-1], 1 / 120)
    s_real = np.fft.fft(x_real)
    s = np.fft.fft(x)
    s_y = np.fft.fft(y)
    w, h = signal.freqs(filt.b, filt.a, freq)

    plt.figure()
    plt.plot(freq, 20 * np.log10(abs(s_real)), label='real')
    plt.plot(freq, 20 * np.log10(abs(s)), label="prediction")
    plt.plot(freq, 20 * np.log10(abs(s_y)), label="filtered")
    plt.plot(w, 20 * np.log10(abs(h)), label='filter')
    plt.legend()
    plt.draw()

    plt.figure()
    plt.plot(x_real, label='real')
    plt.plot(x, label='predict')
    plt.plot(y, label='filtered')
    plt.legend()
    plt.draw()

    plt.show()
