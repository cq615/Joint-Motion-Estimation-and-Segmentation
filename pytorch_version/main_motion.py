import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from network import *
from dataset import *
from util import *
import time
import scipy.io
import os
import matplotlib.pyplot as plt
import pdb


def get_to_cuda(cuda):
    def to_cuda(tensor):
        return tensor.cuda() if cuda else tensor
    return to_cuda


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


def categorical_dice(prediction, truth, k):
    # Dice overlap metric for label value k
    A = (np.argmax(prediction, axis=1) == k)
    B = (np.argmax(truth, axis=1) == k)
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B)+0.001)


def huber_loss(x):
    bsize, csize, height, width = x.size()
    d_x = torch.index_select(x, 3, torch.arange(1, width).cuda()) - torch.index_select(x, 3, torch.arange(width-1).cuda())
    d_y = torch.index_select(x, 2, torch.arange(1, height).cuda()) - torch.index_select(x, 2, torch.arange(height-1).cuda())
    err = torch.sum(torch.mul(d_x, d_x))/height + torch.sum(torch.mul(d_y, d_y))/width
    err /= bsize
    tv_err = torch.sqrt(0.01+err)
    return tv_err


def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False


def plot_grid(gridx, gridy, **kwargs):
    """ plot deformation grid """
    for i in range(gridx.shape[0]):
        plt.plot(gridx[i,:], gridy[i,:], **kwargs)
    for i in range(gridx.shape[1]):
        plt.plot(gridx[:,i], gridy[:,i], **kwargs)


def save_flow(x, pred, x_pred, flow):
    #print(flow.shape)
    x = x.data.cpu().numpy()
    pred = pred.data.cpu().numpy()
    x_pred = x_pred.data.cpu().numpy()
    flow = flow.data.cpu().numpy() * 96
    flow = flow[:,:, 60:140, 40:120]
    X, Y = np.meshgrid(np.arange(0, 80, 2), np.arange(0, 80, 2))
    plt.subplots(figsize=(6, 6))
    plt.subplot(221)
    plt.imshow(x[5, 0, 60:140, 40:120], cmap='gray')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(pred[5, 0, 60:140, 40:120], cmap='gray')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(x_pred[5, 0, 60:140, 40:120], cmap='gray')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(x_pred[5, 0, 60:140, 40:120], cmap='gray')
    plt.quiver(X, Y, flow[5, 0, ::2, ::2], -flow[5, 1, ::2, ::2], scale_units='xy', scale=1, color='r')
    # plot_grid(X - flow[5, 0, ::6, ::6],
    #           Y - flow[5, 1, ::6, ::6],
    #           color='r', linewidth=0.5)
    plt.axis('off')
    plt.savefig('./models/flow_map.png')
    plt.close()


lr = 1e-4
n_worker = 4
bs = 10
n_epoch = 100

model_save_path = './models/model_flow_tmp.pth'

model = Registration_Net()
print(model)
# model.load_state_dict(torch.load(model_save_path))
model = model.cuda()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=lr)
flow_criterion = nn.MSELoss()
Tensor = torch.cuda.FloatTensor


def train(epoch):
    model.train()
    epoch_loss = []
    for batch_idx, batch in tqdm(enumerate(training_data_loader, 1),
                                 total=len(training_data_loader)):
        x, x_pred, x_gnd = batch

        x_c = Variable(x.type(Tensor))
        x_predc = Variable(x_pred.type(Tensor))

        optimizer.zero_grad()
        net = model(x_c, x_predc, x_c)
        flow_loss = flow_criterion(net['fr_st'], x_predc) + 0.01 * huber_loss(net['out'])

        flow_loss.backward()
        optimizer.step()
        epoch_loss.append(flow_loss.item())

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(training_data_loader.dataset),
                100. * batch_idx / len(training_data_loader), np.mean(epoch_loss)))

            save_flow(x_c, x_predc, net['fr_st'], net['out'])
            # scipy.io.savemat(os.path.join('./models/flow_test.mat'),
            #                mdict={'flow': net['out'].data.cpu().numpy()})
            torch.save(model.state_dict(), model_save_path)
            print("Checkpoint saved to {}".format(model_save_path))


data_path = '../test'
train_set = TrainDataset(data_path, transform=data_augment)

# loading the data
training_data_loader = DataLoader(dataset=train_set, num_workers=n_worker,
                                  batch_size=bs, shuffle=True)


for epoch in range(0, n_epoch + 1):

    print('Epoch {}'.format(epoch))

    start = time.time()
    train(epoch)
    end = time.time()
    print("training took {:.8f}".format(end-start))
