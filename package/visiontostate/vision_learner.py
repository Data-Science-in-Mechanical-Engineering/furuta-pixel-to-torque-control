"""
VisionLearner can learn a full_state model from a dataset. The ImagePreprocessor needs to be applied to the dataset
first. The networks are trained with generators for augmentation but the whole dataset gets loaded to RAM. If RAM is
too small code needs to be adjusted to use just with generators (code example can be found in
miscellaneous/visionlearning_first_approach.py).

If modifications are done it is important to check if the normalization and channel order of the images is consistent
through training and prediction! can be done with look_at_data()

DataAugmentation parameters can be adjusted _get_data_generator()

@Author: Steffen Bleher, adapted by Moritz Schneider
"""
import os

import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.nn.parallel import DataParallel
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from visiontostate.dataset import VisionToStateDataset
from visiontostate.models import VisionToStateNet


class VisionLearner:

    def __init__(self, dataID, shuffle, epochs, augmentation, batch_size, val_size,
                 model_path_ext=None):
        my_path = os.path.abspath(os.path.dirname(__file__))
        self.save_path = os.path.join(my_path, '../../data/visiontostate/')
        if model_path_ext is not None:
            self.save_path = os.path.join(self.save_path, model_path_ext)
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path, exist_ok=True)
        self.save_path += str(dataID).zfill(3)
        self.dataID = dataID
        self.val_size = val_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.batch_size = batch_size

        self.epochs = epochs

        self.net = None

    def train(self, parallelize_data=False, load_model=None):
        print("Start training ...")

        # setup to run on gpu
        if torch.cuda.is_available():
            print("Running on gpu ...")
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.net = VisionToStateNet().to(device)

        if load_model is not None:
            print("Loading model...")
            self.net.load_state_dict(torch.load(self.save_path + '/' + load_model + ".pt"))

        run_id = 0
        for (_, dirs, _) in os.walk(self.save_path + '/tb'):
            for dirname in dirs:
                if int(dirname) >= run_id:
                    run_id = int(dirname) + 1
            break
        writer = SummaryWriter(self.save_path + "/tb/" + str(run_id).zfill(3))

        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(1, translate=(0.01, 0.01), scale=(1, 1.02), shear=None, resample=False,
                                    fillcolor=(255, 255, 255)),
            transforms.ColorJitter(brightness=(0.9, 1.1), contrast=0, saturation=0, hue=0),
            transforms.ToTensor()
        ])

        dataset = VisionToStateDataset(root_dir=self.save_path,
                                       transform=data_transform if self.augmentation else None, device=device)


        print("Loaded {} images for training".format(len(dataset)))

        val_samples = int(self.val_size * dataset.__len__())
        train_samples = dataset.__len__() - val_samples

        [train_dataset, val_dataset] = random_split(dataset, [train_samples, val_samples])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=self.shuffle, num_workers=0)

        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                shuffle=self.shuffle, num_workers=0)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=1e-3, amsgrad=True)

        # # Display some images
        # for i_batch, sample_batched in enumerate(val_loader):
        #     for i in range(self.batch_size):
        #         cv2.imshow('original', np.transpose(sample_batched['image'].cpu().numpy()[i], (1, 2, 0))
        #                    .astype(np.float32))
        #         cv2.waitKey(0)
        #         cv2.destroyAllWindows()

        for epoch in range(self.epochs):  # loop over the dataset multiple times
            print("Epoch {} of {}".format(epoch + 1, self.epochs))
            train_loss = 0.0
            train_count = 0
            val_loss = 0.0
            val_count = 0

            # train
            self.net.train()
            for i, data in tqdm(enumerate(train_loader), total=int(train_samples / self.batch_size) + 1):
                optimizer.zero_grad()
                image_batch = data['image']
                label_batch = data['label']
                outputs = self.net(image_batch)
                loss = criterion(outputs, label_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_count += 1
            writer.add_scalar('loss/train', train_loss / train_count, epoch)

            # save model
            torch.save(self.net.state_dict(),
                       self.save_path + "/tb/" + str(run_id).zfill(3) + "/model_epoch{}.pt".format(epoch))

            # validate
            alpha_error = None
            self.net.eval()
            with torch.no_grad():
                for i, data in tqdm(enumerate(val_loader), total=int(val_samples / self.batch_size) + 1):
                    image_batch = data['image']
                    label_batch = data['label']

                    outputs = self.net(image_batch)
                    loss = criterion(outputs, label_batch)
                    val_loss += loss.item()
                    val_count += 1

                    # convergence criteria
                    label_batch = label_batch.detach().cpu().numpy()
                    outputs = outputs.detach().cpu().numpy()

                    alpha = np.arctan2(label_batch[:, 3], label_batch[:, 2])
                    alpha_pred = np.arctan2(outputs[:, 3], outputs[:, 2])

                    combined = np.column_stack((alpha, -alpha_pred))
                    combined = combined[(np.abs(combined[:, 0]) < 10 / 180 * np.pi)]
                    error = np.abs(np.sum(combined, axis=1))

                    if alpha_error is None:
                        alpha_error = error
                    else:
                        alpha_error = np.concatenate((error, alpha_error), axis=0)

            writer.add_scalar('loss/val', val_loss / val_count, epoch)
            writer.add_scalar('loss/val_alpha_error', np.mean(alpha_error), epoch)
            print(alpha_error)
            print(np.mean(alpha_error))
            print(np.mean(alpha_error))
            if np.mean(alpha_error) < 1 / 180 * np.pi:
                print("converged")
                break

        print('Saving model ...')
        torch.save(self.net.state_dict(),
                   self.save_path + "/model{}.pt".format(str(run_id).zfill(3)))

        writer.close()
