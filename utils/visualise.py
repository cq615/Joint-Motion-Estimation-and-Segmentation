import numpy as np
import matplotlib.pyplot as plt
import os, sys


def mask_color_img(img, mask):
    alpha = 0.5
    rows, cols = img.shape
    # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))
    color_mask[mask == 1] = [1, 0, 0]  # Red block
    color_mask[mask == 2] = [0, 1, 0]  # Green block
    color_mask[mask == 3] = [0, 0, 1]  # Blue block

    # Construct RGB version of grey-level image
    img_color = np.dstack((img, img, img))
    img_masked = img_color * 0.8 + np.double(color_mask) * 0.3
    return img_masked

def create_prediction_video(save_dir, img, pred, loc, seq_num):
    # create a video with joint prediction of ROI
    mask = np.argmax(pred, axis=1)
    img_mask_bank = []

    for t in xrange(seq_num):
        img_mask = mask_color_img(img[0, t], mask[t])
        img_mask_bank.append(img_mask)

    mask[mask == 1] = 0
    mask[mask == 3] = 0
    mask[mask == 2] = 1

    mask = np.tile(mask[:, np.newaxis], (1, 2, 1, 1))
    loc = loc * mask
    img_mask_bank = np.array(img_mask_bank)
    flow = loc[:, :, 60:140, 40:120] * 96
    X, Y = np.meshgrid(np.arange(0, 80, 2), np.arange(0, 80, 2))
    for t in xrange(seq_num):
        # meanu = np.mean(flow[t, 0])
        # meanv = np.mean(flow[t, 1])
        plt.imshow(img_mask_bank[t, 60:140, 40:120])

        # mean_scale = np.sqrt(meanu ** 2 + meanv ** 2) * 200
        plt.quiver(X, Y, -flow[t, 0, ::2, ::2], flow[t, 1, ::2, ::2], scale_units='xy', scale=1, color='y')

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, 'test_%d.png'%t))
        plt.close()
    image_dir = os.path.join(save_dir, 'test_%d.png')
    video_dir = os.path.join(save_dir, 'video.avi')
    #os.system('ffmpeg - f image2 - i {0} - vcodec mpeg4 - b 800k {1}'.format(image_dir, video_dir))
    print("Done: images saved in {}".format(image_dir))