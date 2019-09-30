import nibabel as nib
import numpy as np
import os
import h5py
#from utils.util import mkdir
from scipy import ndimage, misc
import cv2


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii.gz"])


def crop_and_fill(img,size):
    img_new = np.zeros((img.shape[0],img.shape[1],size,size))
    h = np.amin([size,img.shape[2]])
    w = np.amin([size,img.shape[3]])
    img_new[:,:,size//2-h//2:size//2+h//2,size//2-w//2:size//2+w//2]=img[:,:,img.shape[2]//2-h//2:img.shape[2]//2+h//2,img.shape[3]//2-w//2:img.shape[3]//2+w//2]
    return img_new


def crop_and_fill_test(img,size):
    img_new = np.zeros((img.shape[0],size,size))
    h = np.amin([size,img.shape[1]])
    w = np.amin([size,img.shape[2]])
    img_new[:,size//2-h//2:size//2+h//2,size//2-w//2:size//2+w//2]=img[:,img.shape[1]//2-h//2:img.shape[1]//2+h//2,img.shape[2]//2-w//2:img.shape[2]//2+w//2]
    return img_new


def load_UKBB_data_3d(data_path, filename, size):
    # Load images and labels, save them into a hdf5 file
    nim = nib.load(os.path.join(data_path, filename, 'sa.nii.gz'))
    image = nim.get_data()[:, :, :, :]

    # generate random index for t and z dimension
    rand_t = np.random.randint(0,image.shape[3])
    #rand_z = np.random.randint(0,image.shape[2])
    rand_z = image.shape[2]//2

    # preprocessing
    image_max = np.max(np.abs(image))
    image /= image_max
    image_sa = image[...,rand_z, rand_t]
    image_sa = image_sa[np.newaxis, np.newaxis]

    frame = np.random.choice(['ED','ES'])

    nim = nib.load(os.path.join(data_path, filename, 'sa_'+frame+'.nii.gz'))
    image = nim.get_data()[:, :, :]

    nim_seg = nib.load(os.path.join(data_path, filename, 'label_sa_'+frame+'.nii.gz'))
    seg = nim_seg.get_data()[:, :, :]

    image_frame = image[...,rand_z]
    image_frame /= image_max
    seg_frame = seg[...,rand_z]

    image_frame = image_frame[np.newaxis, np.newaxis]
    seg_frame = seg_frame[np.newaxis, np.newaxis]

    image_bank = np.concatenate((image_sa, image_frame), axis=1)

    image_bank = crop_and_fill(image_bank, size)
    seg_bank = crop_and_fill(seg_frame, size)
    image_bank = np.transpose(image_bank, (0, 1, 3, 2))
    seg_bank = np.transpose(seg_bank, (0, 1, 3, 2))
    image_bank = np.array(image_bank, dtype='float32')
    seg_bank = np.array(seg_bank, dtype='int16')
    return image_bank, seg_bank


def data_augment(image, label, shift=10.0, rotate=10.0, scale=0.1, intensity=0.1, flip=False):
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


def load_dataset(filename):
    f = h5py.File(filename, 'r')
    image = np.array(f['image'])
    label = np.array(f['label'])
    return image, label


def load_UKBB_test_data(data_path, filename, frame, size):
    nim = nib.load(os.path.join(data_path, filename, 'sa.nii.gz'))
    image = nim.get_data()[:, :, :, :]
    dx = nim.header.get_zooms()

    # print(image.shape)
    image = np.array(image, dtype='float32')
    image_max = np.max(np.abs(image))

    image_bank = []
    seg_bank = []
    nim = nib.load(os.path.join(data_path, filename, 'sa_' + frame + '.nii.gz'))
    image = nim.get_data()[:, :, :]
    image /= image_max
    image = np.array(image, dtype='float32')

    nim_seg = nib.load(os.path.join(data_path, filename, 'label_sa_' + frame + '.nii.gz'))
    seg = nim_seg.get_data()[:, :, :]

    image = np.transpose(image, (2, 0, 1))
    seg = np.transpose(seg, (2, 0, 1))
    image_bank.append(image)
    seg_bank.append(seg)

    image_bank = np.concatenate(image_bank)
    seg_bank = np.concatenate(seg_bank)

    image_bank = crop_and_fill_test(image_bank, size)
    seg_bank = crop_and_fill_test(seg_bank, size)
    image_bank = np.transpose(image_bank, (0, 2, 1))
    seg_bank = np.transpose(seg_bank, (0, 2, 1))

    image_bank = np.array(image_bank, dtype='float32')
    seg_bank = np.array(seg_bank, dtype='int32')

    return image_bank, seg_bank, dx
