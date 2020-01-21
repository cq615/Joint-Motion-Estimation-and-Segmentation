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


def save_flow(flow):
    #print(flow.shape)
    flow = flow * 96
    X, Y = np.meshgrid(np.linspace(0, 192, 32), np.linspace(0, 192, 32))
    plt.quiver(X, Y, -flow[5, 0, ::6, ::6], -flow[5, 1, ::6, ::6], scale_units='xy', scale=1)
    plt.axis('off')
    plt.savefig('./models/flow_map.png')
    plt.close()

lr = 1e-4
n_class = 4
n_worker = 4
bs = 10
n_epoch = 100

model_save_path = './models/joint_model_tmp.pth'

model = Seg_Motion_Net()
model = model.cuda()
# model.load_state_dict(torch.load(model_save_path), strict=False)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=lr)
flow_criterion = nn.MSELoss()
seg_criterion = nn.CrossEntropyLoss()
Tensor = torch.cuda.FloatTensor


def train(epoch):
    model.train()
    epoch_loss = []
    LV_dice = 0
    Myo_dice = 0
    RV_dice = 0
    for batch_idx, batch in tqdm(enumerate(training_data_loader, 1),
                                 total=len(training_data_loader)):
        x, x_pred, x_gnd = batch
        x_c = Variable(x.type(Tensor))
        x_predc = Variable(x_pred.type(Tensor))
        x_gndc = Variable(x_gnd.type(torch.cuda.LongTensor))

        optimizer.zero_grad()
        net = model(x_c, x_predc, x_c)
        flow_loss = flow_criterion(net['fr_st'], x_predc) + 0.01 * huber_loss(net['out'])
        seg_loss = seg_criterion(net['outs_softmax'], x_gndc)
        loss = flow_loss + 0.01 * seg_loss
        flow_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        seg_loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        pred = net['outs_softmax'].data.cpu().numpy()
        truth = convert_to_1hot(x_gnd[:,None].numpy(), n_class)
        LV_dice += categorical_dice(pred, truth, 1)
        Myo_dice += categorical_dice(pred, truth, 2)
        RV_dice += categorical_dice(pred, truth, 3)
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(training_data_loader.dataset),
                100. * batch_idx / len(training_data_loader), np.mean(epoch_loss)))
            print("  Training Dice LV:\t\t{:.6f}".format(LV_dice / batch_idx))
            print("  Training Dice Myo:\t\t{:.6f}".format(Myo_dice / batch_idx))
            print("  Training Dice RV:\t\t{:.6f}".format(RV_dice / batch_idx))
            # pdb.set_trace()
            # save_flow(net['out'].data.cpu().numpy())
            # scipy.io.savemat(os.path.join('./models/flow_test.mat'),
            #                mdict={'flow': net['out'].data.cpu().numpy()})
            torch.save(model.state_dict(), model_save_path)
            print("Checkpoint saved to {}".format(model_save_path))


def test():
    # test the segmentation performance
    model.eval()
    epoch_loss = []
    LV_dice = 0
    Myo_dice = 0
    RV_dice = 0
    test_batches = 0

    for batch_idx, batch in tqdm(enumerate(testing_data_loader, 1),
                                 total=len(testing_data_loader)):
        x, x_gnd = batch
        x = x.permute(1,0,2,3)
        x_gnd = x_gnd.permute(1,0,2,3)
        with torch.no_grad():
            x_c = Variable(x).cuda()
            x_gndc = Variable(x_gnd[:,0]).cuda().long()

        net = model(x_c, x_c, x_c)

        seg_loss = seg_criterion(net['outs_softmax'], x_gndc)
        pred = net['outs_softmax'].data.cpu().numpy()
        truth = convert_to_1hot(x_gnd.numpy(), n_class)
        LV_dice += categorical_dice(pred, truth, 1)
        Myo_dice += categorical_dice(pred, truth, 2)
        RV_dice += categorical_dice(pred, truth, 3)
        test_batches += 1
        epoch_loss.append(seg_loss.item())
        print('\nTest set: Average loss: {:.4f}\n'.format(np.mean(epoch_loss)))
        print("  testing Dice LV:\t\t{:.6f}".format(LV_dice/test_batches))
        print("  testing Dice Myo:\t\t{:.6f}".format(Myo_dice/test_batches))
        print("  testing Dice RV:\t\t{:.6f}".format(RV_dice/test_batches))


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

    for frame in ['ED', 'ES']:
        test_set = TestDataset(data_path, frame)
        testing_data_loader = DataLoader(dataset=test_set, num_workers=n_worker,
                                         batch_size=1, shuffle=False)
        test()


