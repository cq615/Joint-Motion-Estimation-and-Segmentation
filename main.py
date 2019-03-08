import os, time, h5py, sys
from models.seg_network import build_FCN_triple_branch
from dataio.dataset import *
from utils.metrics import *
from dataio.data_generator import *
from utils.visualise import *
import lasagne.layers as L
import lasagne
import theano
import theano.tensor as T

if __name__ == '__main__':

    shift = 10
    rotate = 10
    scale = 0.1
    intensity = 0.1
    flip = False

    base_path = 'seg_flow'
    data_path = os.path.join(base_path, 'data')
    model_path = os.path.join(base_path, 'model')
    size = 192
    n_class = 4

    # Prepare theano variables

    image_var = T.tensor4('image')
    image_pred_var = T.tensor4('image_pred')
    label_var = T.itensor4('label')
    image_seg_var = T.tensor4('seg')

    # Compile the model (CNN one, to build RNN one, compile build_FCN_triple_branch_rnn and use sequence data)
    net = build_FCN_triple_branch(image_var, image_pred_var, image_seg_var, shape=(None, 1, size, size))
    model_file = os.path.join(model_path, 'FCN_VGG16_sz192_joint_learn_ft_tmp.npz')
    with np.load(model_file) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    L.set_all_param_values([net['out'], net['outs']], param_values)

    learning_rate = T.scalar('learning_rate')
    prediction = L.get_output(net['fr_st'])
    loc = L.get_output(net['out'])
    flow_loss = lasagne.objectives.squared_error(prediction, image_pred_var)
    flow_loss = flow_loss.mean() + 0.001 * huber_loss(loc)

    prediction_seg = L.get_output(net['outs'])
    loss_seg = categorical_crossentropy(prediction_seg, label_var)
    loss = flow_loss + 0.01 * loss_seg.mean()

    params = L.get_all_params(net['out'], trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)

    # Compile a function performing a training step on a mini-batch and returning the corresponding training loss
    train_fn_flow = theano.function([image_var, image_pred_var, image_seg_var, label_var, learning_rate],
                                    [loss, flow_loss, loss_seg.mean()], updates=updates, on_unused_input='ignore')

    params = lasagne.layers.get_all_params(net['outs'], trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
    train_fn_seg = theano.function([image_var, image_pred_var, image_seg_var, label_var, learning_rate], [loss,flow_loss, loss_seg.mean()], updates=updates,on_unused_input='ignore')

    model_name = 'FCN_VGG16_sz{0}'.format(size)
    test_prediction = lasagne.layers.get_output(net['outs'])
    test_loss = categorical_crossentropy(test_prediction, label_var)

    test_acc = categorical_accuracy(test_prediction, label_var)
    test_dice_lv = categorical_dice(test_prediction, label_var, 1)
    test_dice_myo = categorical_dice(test_prediction, label_var, 2)
    test_dice_rv = categorical_dice(test_prediction, label_var, 3)

    # Compile a second function computing the testing loss and accuracy
    test_fn = theano.function([image_seg_var, label_var], [test_acc, test_dice_lv, test_dice_myo, test_dice_rv],
                              on_unused_input='ignore')

    # Launch the training loop
    print("Starting training...")
    np.random.seed(100)
    start_time = time.time()
    table = []
    num_epoch = 100
    max_iter = 100
    batch_size = 64
    start = 1
    lr = 1e-4
    log_every = 100

    data_train_path = 'train'
    test_image, test_label = load_dataset(os.path.join(data_path, 'test_UKBB2964_sz{0}_n100.h5'.format(size)))
    for epoch in range(start, start + num_epoch):
        # In each epoch, we do a full pass over the training data:
        start_time_epoch = time.time()
        train_loss = 0
        train_loss_image = 0
        train_loss_seg = 0
        train_batches = 10

        for t in range(max_iter):
            image, label = generate_batch(data_train_path, batch_size=6, img_size=192)
            image2, label2 = data_augmenter(image, label, shift=shift, rotate=rotate, scale=scale, intensity=intensity,
                                            flip=flip)
            label2_1hot = convert_to_1hot(label2, n_class)
            # Train motion estimation and segmentation iteratively. After pretraining, the network can be trained jointly.
            loss, loss_flow, loss_seg = train_fn_flow(image2[:, 0:1], image2[:, 1:], image2[:, 1:], label2_1hot, lr)
            loss, loss_flow, loss_seg = train_fn_seg(image2[:,0:1], image2[:,1:],image2[:,1:], label2_1hot, lr)
            
            # add warped segmentation loss
            #loss, loss_flow, loss_seg = train_fn_seg_warped(image2[:, 0:1], image2[:, 1:], image2[:, 0:1], label2_1hot,1e-6)

            train_loss += loss
            train_loss_image += loss_flow
            train_loss_seg += loss_seg
            train_batches += 1

            if train_batches % log_every == 0:

                #
                # And a full pass over the testing data:
                test_loss = 0
                test_acc = 0
                test_dice_lv = 0
                test_dice_myo = 0
                test_dice_rv = 0
                test_batches = 0
                for image_test, label_test in iterate_minibatches(test_image, test_label, batch_size):
                    label_1hot = convert_to_1hot(label_test, n_class)

                    acc, dice_lv, dice_myo, dice_rv = test_fn(image_test, label_1hot)

                    test_acc += acc
                    test_dice_lv += dice_lv
                    test_dice_myo += dice_myo
                    test_dice_rv += dice_rv
                    test_batches += 1

                test_acc /= test_batches
                test_dice_lv /= test_batches
                test_dice_myo /= test_batches
                test_dice_rv /= test_batches

                # Then we print the results for this epoch:
                print("Epoch {} of {} took {:.3f}s".format(epoch, start + num_epoch - 1, time.time() - start_time_epoch))
                # print('  learning rate:\t\t{:.8f}'.format(float(learning_rate.get_value())))
                print("  training loss:\t\t{:.6f}".format(train_loss / train_batches))
                print("  training loss flow: \t\t{:.6f}".format(train_loss_image / train_batches))
                print("  training loss seg: \t\t{:.6f}".format(train_loss_seg / train_batches))
                print("  testing accuracy:\t\t{:.2f} %".format(test_acc * 100))
                print("  testing Dice LV:\t\t{:.6f}".format(test_dice_lv))
                print("  testing Dice Myo:\t\t{:.6f}".format(test_dice_myo))
                print("  testing Dice RV:\t\t{:.6f}".format(test_dice_rv))


                np.savez(os.path.join(model_path, '{0}_joint_learn_ft_tmp.npz'.format(model_name, epoch)),
                         *lasagne.layers.get_all_param_values([net['out'], net['outs']]))

        np.savez(os.path.join(model_path, '{0}_joint_learn_ft1_epoch{1:03d}.npz'.format(model_name, epoch)),
                 *lasagne.layers.get_all_param_values([net['out'], net['outs']]))

    print("Training took {:.3f}s in total.".format(time.time() - start_time))