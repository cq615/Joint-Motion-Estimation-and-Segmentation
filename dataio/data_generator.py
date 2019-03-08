import os, h5py,sys
import nibabel as nib, numpy as np

# A generator function using the keyword yield
def iterate_minibatches(image, label, batch_size):
    assert len(image) == len(label)
    for start_idx in range(0, len(image) - batch_size + 1, batch_size):
        end_idx = start_idx + batch_size
        idx = slice(start_idx, end_idx)
        yield image[idx], label[idx]

def crop_and_fill(img,size):
    img_new = np.zeros((img.shape[0],img.shape[1],size,size))
    h = np.amin([size,img.shape[2]])
    w = np.amin([size,img.shape[3]])
    img_new[:,:,size//2-h//2:size//2+h//2,size//2-w//2:size//2+w//2]=img[:,:,img.shape[2]//2-h//2:img.shape[2]//2+h//2,img.shape[3]//2-w//2:img.shape[3]//2+w//2]
    return img_new


def load_UKBB_train_data_3d(data_path, filename, size):
    # Load images and labels, save them into a hdf5 file
    nim = nib.load(os.path.join(data_path, filename, 'sa.nii.gz'))
    image = nim.get_data()[:, :, :, :]

    # generate random index for t and z dimension
    rand_t = np.random.randint(0,image.shape[3])
    rand_z = np.random.randint(0,image.shape[2])

    # preprocessing
    image_max = np.max(np.abs(image))
    image /= image_max
    image_sa = image[...,rand_z, rand_t]
    image_sa = image_sa[np.newaxis, np.newaxis]

    # Read image and header information
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
    # plt.subplot(211)
    # plt.imshow(image_bank[0,1],cmap='gray')
    # plt.subplot(212)
    # plt.imshow(seg_bank[0,0],cmap='gray')
    # plt.show()

    return image_bank, seg_bank

def generate_batch(data_path, batch_size, img_size):
    filename = [f for f in sorted(os.listdir(data_path))]
    batch_num = np.random.randint(0,len(os.listdir(data_path)), batch_size)
    batch_filename = [filename[i] for i in batch_num]
    train_image=[]
    train_label=[]
    for f in batch_filename:
        img, seg = load_UKBB_train_data_3d(data_path, f, img_size)

        train_image.append(img)
        train_label.append(seg)
    train_image = np.concatenate(train_image)
    train_label = np.concatenate(train_label)
    return train_image, train_label


def load_UKBB_test_data_3d(data_path, filename, size):
    # load UKBB dataset (only the mid-ventricular slice)
    nim = nib.load(os.path.join(data_path, filename, 'sa.nii.gz'))
    image = nim.get_data()[:, :, :, :]

    image_mid = image[:, :, int(round(image.shape[2] // 2.0))]
    image_mid = np.array(image_mid, dtype='float32')

    # preprocessing
    curr_data = image_mid
    pl, ph = np.percentile(curr_data, (.01, 99.99))
    curr_data[curr_data < pl] = pl
    curr_data[curr_data > ph] = ph
    curr_data = (curr_data.astype(float) - pl) / (ph - pl)
    image_mid = curr_data

    nim_seg = nib.load(os.path.join(data_path, filename, 'label_sa_ED.nii.gz'))
    seg = nim_seg.get_data()[:, :, :]
    seg_mid = seg[:, :, int(round(seg.shape[2] // 2.0))]
    seg_mid = seg_mid[np.newaxis]

    image_bank = image_mid[np.newaxis]

    seg_bank = seg_mid[..., np.newaxis]

    image_bank = np.transpose(image_bank, (0, 3, 2, 1))
    seg_bank = np.transpose(seg_bank, (0, 3, 2, 1))
    image_bank = crop_and_fill(image_bank, size)
    seg_bank = crop_and_fill(seg_bank, size)
    seg_bank = np.array(seg_bank, dtype='int32')
    image_bank = np.array(image_bank, dtype='float32')
    return image_bank, seg_bank



