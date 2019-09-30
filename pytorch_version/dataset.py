import torch.utils.data as data
import numpy as np
import datetime

from os import listdir
from os.path import join
from util import *


class TrainDataset(data.Dataset):
    def __init__(self, data_path, transform=None):
        super(TrainDataset, self).__init__()
        self.data_path = data_path
        self.filename = [f for f in sorted(listdir(self.data_path))]

        # data augmentation
        self.transform = transform

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images

        input, target = load_UKBB_data_3d(self.data_path, self.filename[index], size = 192)

        if self.transform:
            input, target = self.transform(input, target)

        image = input[0,:1]
        image_pred = input[0,1:]

        return image, image_pred, target[0,0]

    def __len__(self):
        return len(self.filename)


class TestDataset(data.Dataset):
    def __init__(self, data_path, frame, transform=None):
        super(TestDataset, self).__init__()
        self.data_path = data_path
        self.filename = [f for f in sorted(listdir(self.data_path))]

        # data augmentation
        self.transform = transform
        self.frame = frame

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        #np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images

        input, target, dx = load_UKBB_test_data(self.data_path, self.filename[index], self.frame, size = 192)

        if self.transform:
            input, target = self.transform(input, target)

        return input, target

    def __len__(self):
        return len(self.filename)


class TestDataset_flow(data.Dataset):
    def __init__(self, data_path, transform=None):
        super(TestDataset_flow, self).__init__()
        self.data_path = data_path
        self.filename = [f for f in sorted(listdir(self.data_path))]

        # data augmentation
        self.transform = transform

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        #np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images

        input_ed, target_ed, dx = load_UKBB_test_data(self.data_path, self.filename[index], 'ED', size=192)
        input_es, target_es, dx = load_UKBB_test_data(self.data_path, self.filename[index], 'ES', size=192)

        # if self.transform:
        #     input, target = self.transform(input, target)

        return input_ed, target_ed, input_es, target_es, dx

    def __len__(self):
        return len(self.filename)
