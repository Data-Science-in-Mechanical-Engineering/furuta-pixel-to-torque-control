"""
Simple Dataset for VisionLearner

@Author: Steffen Bleher
"""

from __future__ import print_function, division

import os

import cv2
import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset

class VisionToStateDataset(Dataset):

    def __init__(self, root_dir, transform=None, device=torch.device("cpu")):
        """
        Args:
            root_dir (string): 00x directory in data/visiontostate.
        """
        self.angle_data = pd.read_csv(root_dir + '/angles.csv', header=None, error_bad_lines=False).iloc[:, 0:5]
        self.root_dir = root_dir
        self.transform = transform
        self.device = device

        # # if smaller dataset needed for testing
        # self.angle_data = self.angle_data.iloc[0:1000, :]

    def __len__(self):
        return len(self.angle_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        angle_data = self.angle_data.iloc[idx, 1:]
        angle_data = np.array([angle_data])
        #angle_data = torch.tensor(angle_data.astype('float').reshape(4), device=self.device).float()
        angle_data = torch.tensor(angle_data.astype('float').reshape(4), dtype=torch.float, device=self.device)

        img_name = os.path.join(self.root_dir + "/img", self.angle_data.iloc[idx, 0])
        image = io.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self.transform:
            image = self.transform(image).float()
        else:
            #image = torch.tensor(image, device=self.device).float()
            image = torch.tensor(image, dtype=torch.float, device=self.device)

        sample = {'image': image, 'label': angle_data}

        return sample