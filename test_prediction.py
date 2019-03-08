"""
Created on Mon Feb 12 17:52:19 2018

@author: cq615
"""
import os, time, h5py, sys
from models.seg_network import build_FCN_triple_branch_rnn
from dataio.dataset import *
from utils.metrics import *
from dataio.data_generator import *
from utils.visualise import *
import lasagne.layers as L
import matplotlib.pyplot as plt
import theano
import theano.tensor as T


def tensor5(name=None, dtype=None):
    if dtype is None:
        dtype = theano.config.floatX
    type = T.TensorType(dtype, (False, )*5)
    return type(name)


if __name__ == '__main__':

    data_test_path = 'test'
    save_dir = 'visualisation'

    seq_num = 50
    n_class = 4
    size = 192

    # Build the network

    image_var = tensor5('image')
    image_pred_var = tensor5('image_pred')
    label_var = T.itensor4('label')
    image_seg_var = T.tensor4('image_seg')
    
    net = build_FCN_triple_branch_rnn(image_var, image_pred_var, image_seg_var, shape = (None,1,size, size, seq_num), shape_seg = (None, 1, size, size)) 
    #model_file = 'model/FCN_VGG16_sz192_flow_simese_rnn_shared.npz'
    model_file = 'model/FCN_VGG16_sz192_triple_3d_rnn_warped_tmp.npz'
    with np.load(model_file) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    L.set_all_param_values([net['out'],net['outs']], param_values)

    test_prediction = L.get_output(net['outs'])
    test_loc = L.get_output(net['out'], deterministic = True)
    test_fn = theano.function([image_var, image_pred_var, image_seg_var], [test_prediction, test_loc], on_unused_input='ignore')

    filename = [f for f in sorted(os.listdir(data_test_path))]
    for f_id in filename[1:2]:
        # Load the dataset
        print("Loading data...")
        img, seg = load_UKBB_test_data_3d(data_test_path, f_id, size)

        print("Starting testing...")
        # reshape input
        input_data_seg = np.reshape(img,(-1,size,size))
        input_data_seg = input_data_seg[:,np.newaxis]
        input_data = np.transpose(img,(0,2,3,1))
        input_data = np.expand_dims(input_data, axis=1)

        input_data_pred = np.tile(input_data[...,0:1],(1,1,1,1,input_data.shape[4]))
        start_time = time.time()
        pred, loc = test_fn(input_data_pred, input_data, input_data_seg)
        print("Multi-task prediction time:\t{:.2f}s".format(time.time()-start_time))
        print("Creating a video for joint prediction...")
        create_prediction_video(save_dir, img, pred, loc, seq_num)



