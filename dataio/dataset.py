import os, time, h5py, sys
import nibabel as nib
import cv2
import numpy as np
from scipy import ndimage


def load_dataset(filename):
    f = h5py.File(filename, 'r')
    image = np.array(f['image'])
    label = np.array(f['label'])
    return image, label


def convert_to_1hot(label, n_class):
    # Convert a label map (N x 1 x H x W) into a one-hot representation (N x C x H x W)
    label_swap = label.swapaxes(1, 3)
    label_flat = label_swap.flatten()
    n_data = len(label_flat)
    label_1hot = np.zeros((n_data, n_class), dtype='int16')
    label_1hot[range(n_data), label_flat] = 1
    label_1hot = label_1hot.reshape((label_swap.shape[0], label_swap.shape[1], label_swap.shape[2], n_class))
    label_1hot = label_1hot.swapaxes(1, 3)
    return label_1hot


def convert_to_1hot_3d(label, n_class):
    # Convert a label map (N1XYZ) into a one-hot representation (NCXYZ)
    label_swap = label.swapaxes(1, 4)
    label_flat = label_swap.flatten()
    n_data = len(label_flat)
    label_1hot = np.zeros((n_data, n_class), dtype='int16')
    label_1hot[range(n_data), label_flat] = 1
    label_1hot = label_1hot.reshape((label_swap.shape[0], label_swap.shape[1], label_swap.shape[2], label_swap.shape[3], n_class))
    label_1hot = label_1hot.swapaxes(1, 4)
    return label_1hot

def data_augmenter(image, label, shift=0.0, rotate=0.0, scale=0.0, intensity=0.0, flip=False):
    # Perform affine transformation on image and label, which are 4D tensors of dimension (N, C, X, Y).
    image2 = np.zeros(image.shape, dtype='float32')
    label2 = np.zeros(label.shape, dtype='int16')
    for i in range(image.shape[0]):
        # Random affine transformation using normal distributions
        shift_var = [np.clip(np.random.normal(), -3, 3) * shift, np.clip(np.random.normal(), -3, 3) * shift]
        rotate_var = np.clip(np.random.normal(), -3, 3) * rotate
        scale_var = 1 + np.clip(np.random.normal(), -3, 3) * scale
        intensity_var = 1 + np.clip(np.random.normal(), -3, 3) * intensity

        # Apply affine transformation (rotation + scale + shift) to training images
        row, col = image.shape[2:]
        M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_var, 1.0 / scale_var)
        M[:, 2] += shift_var
        for c in range(image.shape[1]):
            image2[i, c] = ndimage.interpolation.affine_transform(image[i, c], M[:, :2], M[:, 2], order=1)
        label2[i, 0] = ndimage.interpolation.affine_transform(label[i, 0], M[:, :2], M[:, 2], order=0)

        # Apply intensity variation
        image2[i, :] *= intensity_var

        # Apply random horizontal or vertical flipping
        if flip:
            if np.random.uniform() >= 0.5:
                image2[i, :] = image2[i, :, ::-1, :]
                label2[i, 0] = label2[i, 0, ::-1, :]
            else:
                image2[i, :] = image2[i, :, :, ::-1]
                label2[i, 0] = label2[i, 0, :, ::-1]
    return image2, label2


def data_augmenter_3d(image, label, shift=0.0, rotate=0.0, scale=0.0, intensity=0.0, flip=False):
    # Perform affine transformation on image and label, which are 4D tensors of dimension (N, C, X, Y).
    image2 = np.zeros(image.shape, dtype='float32')
    label2 = np.zeros(label.shape, dtype='int16')
    for i in range(image.shape[0]):
        # Random affine transformation using normal distributions
        shift_var = [np.clip(np.random.normal(), -3, 3) * shift, np.clip(np.random.normal(), -3, 3) * shift]
        rotate_var = np.clip(np.random.normal(), -3, 3) * rotate
        scale_var = 1 + np.clip(np.random.normal(), -3, 3) * scale
        intensity_var = 1 + np.clip(np.random.normal(), -3, 3) * intensity

        # Apply affine transformation (rotation + scale + shift) to training images
        row, col = image.shape[2:4]
        M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_var, 1.0 / scale_var)
        M[:, 2] += shift_var
        for z in range(image.shape[4]):
            image2[i, 0, :, :, z] = ndimage.interpolation.affine_transform(image[i, 0, :, :, z], M[:, :2], M[:, 2], order=1)
            label2[i, 0, :, :, z] = ndimage.interpolation.affine_transform(label[i, 0, :, :, z], M[:, :2], M[:, 2], order=0)

        # Apply intensity variation
        image2[i] *= intensity_var

        # Apply random horizontal or vertical flipping
        if flip:
            if np.random.uniform() >= 0.5:
                image2[i] = image2[i, :, ::-1, :]
                label2[i] = label2[i, :, ::-1, :]
            else:
                image2[i] = image2[i, :, :, ::-1]
                label2[i] = label2[i, :, :, ::-1]
    return image2, label2
