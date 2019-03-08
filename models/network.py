# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:44:08 2018

@author: cq615
"""

import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn
import lasagne
import lasagne.layers as L
from lasagne.layers.base import Layer
from lasagne.utils import as_tuple
from lasagne.layers.pool import pool_output_length
import numpy as np
from layers import *

def Conv(incoming, num_filters, filter_size=3,
         stride=(1, 1), pad='same', W=lasagne.init.HeNormal(),
         b=None, nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
    """
    Overrides the default parameters for ConvLayer
    """
    ensure_set_name('conv', kwargs)

    return L.Conv2DLayer(incoming, num_filters, filter_size, stride, pad, W=W, b=b,
                     nonlinearity=nonlinearity, **kwargs)
# Calculate the memory required for a network
def memory_requirement(layer):
    # Data blobs
    sum = 0
    for l in L.get_all_layers(layer):
        sum += np.prod(l.output_shape)
    float_len = 4
    sum = (sum * float_len) / pow(2, 30)
    print('Memory for data blobs = {0:.3f}GB'.format(sum))

    # Parameters
    sum = 0
    for l in L.get_all_layers(layer):
        for p in l.get_params():
            sum += np.prod(p.get_value().shape)
    print('Number of parameters  = {0}'.format(sum))
    sum = (sum * float_len) / pow(2, 30)
    print('Memory for parameters = {0:.3f}GB'.format(sum))


def bilinear_1d(sz):
    if sz % 2 == 0:
        raise NotImplementedError('`Bilinear kernel` requires odd filter size.')
    c = (sz + 1) / 2
    h = np.array(range(1, c + 1) + range(c - 1, 0, -1))
    h = h / float(c)
    return h


def bilinear_2d(sz):
    W = np.ones((sz, sz))
    h = bilinear_1d(sz)

    for i in range(sz):
        W[i, :] *= h

    for j in range(sz):
        W[:, j] *= h
    return W


def set_conv_bilinear_weights(params, num_filters, filter_size):
    # Set weights
    [W, b] = params
    W_val = np.zeros((num_filters, num_filters, filter_size, filter_size), dtype=np.float32)
    for c in range(num_filters):
        W_val[c, c, :, :] = bilinear_2d(filter_size)
    b_val = np.zeros((num_filters,), dtype=np.float32)
    W.set_value(W_val)
    b.set_value(b_val)


class BilinearUpsamplingLayer(Layer):
    """
    2D bilinear upsampling layer

    Performs 2D bilinear upsampling over the two trailing axes of a 4D or 5D input tensor.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    scale_factor : integer or iterable
        The scale factor in each dimension. If an integer, it is promoted to
        a square scale factor region. If an iterable, it should have two
        elements.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incoming, scale_factor, **kwargs):
        super(BilinearUpsamplingLayer, self).__init__(incoming, **kwargs)

        self.scale_factor = scale_factor
        if self.scale_factor < 1:
            raise ValueError('Scale factor must be >= 1, not {0}'.format(self.scale_factor))

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list
        if output_shape[2] is not None:
            output_shape[2] *= self.scale_factor
        if output_shape[3] is not None:
            output_shape[3] *= self.scale_factor
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        upscaled = input
        if self.scale_factor > 1:
            if input.ndim == 4:
                upscaled = T.nnet.abstract_conv.bilinear_upsampling(input=input, ratio=self.scale_factor)
            elif input.ndim == 5:
                # Swap dimension order from bcxyz to bczyx
                input_swap = input.swapaxes(2, 4)
                shape = input_swap.shape
                # Squeeze the first two dimensions so it becomes a 4D tensor
                # and 2D bilinear_upsampling can be applied.
                input_reshape = input_swap.reshape((shape[0] * shape[1], shape[2], shape[3], shape[4]))
                upscaled = T.nnet.abstract_conv.bilinear_upsampling(input=input_reshape, ratio=self.scale_factor)
                # Recover the 5D tensor shape
                upscaled_reshape = upscaled.reshape((shape[0], shape[1], shape[2], \
                                                     shape[3] * self.scale_factor, shape[4] * self.scale_factor))
                upscaled = upscaled_reshape.swapaxes(2, 4)
        return upscaled


class Conv3DDNNLayer(lasagne.layers.conv.BaseConvLayer):
    """
    lasagne.layers.Conv3DDNNLayer(incoming, num_filters, filter_size,
    stride=(1, 1, 1), pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, flip_filters=False,
    **kwargs)

    3D convolutional layer

    Performs a 3D convolution on its input and optionally adds a bias and
    applies an elementwise nonlinearity.  This implementation uses
    ``theano.sandbox.cuda.dnn.dnn_conv3d`` directly.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 5D tensor, with shape ``(batch_size,
        num_input_channels, input_rows, input_columns, input_depth)``.

    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable of int
        An integer or a 3-element tuple specifying the size of the filters.

    stride : int or iterable of int
        An integer or a 3-element tuple specifying the stride of the
        convolution operation.

    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of three integers allows different symmetric
        padding per dimension.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.

        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).

        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.

    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        4D tensor.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 5D tensor with shape ``(num_filters,
        num_input_channels, filter_rows, filter_columns, filter_depth)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, output_rows, output_columns, output_depth)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    flip_filters : bool (default: False)
        Whether to flip the filters and perform a convolution, or not to flip
        them and perform a correlation. Flipping adds a bit of overhead, so it
        is disabled by default. In most cases this does not make a difference
        anyway because the filters are learned, but if you want to compute
        predictions with pre-trained weights, take care if they need flipping.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.
    """
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1, 1),
                 pad=0, untie_biases=False,
                 W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify, flip_filters=False,
                 convolution=dnn.dnn_conv3d,
                 **kwargs):
        super(Conv3DDNNLayer, self).__init__(incoming, num_filters, filter_size,
                                             stride, pad, untie_biases, W, b,
                                             nonlinearity, flip_filters, n=3,
                                             **kwargs)
        self.convolution = convolution

    def convolve(self, input, **kwargs):
        # by default we assume 'cross', consistent with corrmm.
        conv_mode = 'conv' if self.flip_filters else 'cross'
        border_mode = self.pad
        if border_mode == 'same':
            border_mode = tuple(s // 2 for s in self.filter_size)

        conved = self.convolution(img=input, kerns=self.W,
                                  border_mode=border_mode,
                                  subsample=self.stride,
                                  conv_mode=conv_mode)
        return conved


class Pool3DDNNLayer(Layer):
    """
    3D pooling layer

    Performs 3D mean- or max-pooling over the 3 trailing axes of a 5D input
    tensor. This is an alternative implementation which uses
    ``theano.sandbox.cuda.dnn.dnn_pool`` directly.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_size : integer or iterable
        The length of the pooling region in each dimension. If an integer, it
        is promoted to a square pooling region. If an iterable, it should have
        two elements.

    stride : integer, iterable or ``None``
        The strides between sucessive pooling regions in each dimension.
        If ``None`` then ``stride = pool_size``.

    pad : integer or iterable
        Number of elements to be added on each side of the input
        in each dimension. Each value must be less than
        the corresponding stride.

    ignore_border : bool (default: True)
        This implementation never includes partial pooling regions, so this
        argument must always be set to True. It exists only to make sure the
        interface is compatible with :class:`lasagne.layers.MaxPool2DLayer`.

    mode : string
        Pooling mode, one of 'max', 'average_inc_pad' or 'average_exc_pad'.
        Defaults to 'max'.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    The value used to pad the input is chosen to be less than
    the minimum of the input, so that the output of each pooling region
    always corresponds to some element in the unpadded input region.

    """
    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0, 0),
                 ignore_border=True, mode='max', **kwargs):
        super(Pool3DDNNLayer, self).__init__(incoming, **kwargs)

        self.pool_size = as_tuple(pool_size, 3)

        if len(self.input_shape) != 5:
            raise ValueError("Tried to create a 3D pooling layer with "
                             "input shape %r. Expected 5 input dimensions "
                             "(batchsize, channels, 3 spatial dimensions)."
                             % (self.input_shape,))

        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = as_tuple(stride, 3)

        self.pad = as_tuple(pad, 3)
        self.mode = mode

        # The ignore_border argument is for compatibility with MaxPool2DLayer.
        # ignore_border=False is not supported. Borders are always ignored.
        self.ignore_border = ignore_border
        if not self.ignore_border:
            raise NotImplementedError("Pool3DDNNLayer does not support "
                                      "ignore_border=False.")

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[2] = pool_output_length(input_shape[2],
                                             pool_size=self.pool_size[0],
                                             stride=self.stride[0],
                                             pad=self.pad[0],
                                             ignore_border=self.ignore_border,
                                             )

        output_shape[3] = pool_output_length(input_shape[3],
                                             pool_size=self.pool_size[1],
                                             stride=self.stride[1],
                                             pad=self.pad[1],
                                             ignore_border=self.ignore_border,
                                             )

        output_shape[4] = pool_output_length(input_shape[4],
                                             pool_size=self.pool_size[2],
                                             stride=self.stride[2],
                                             pad=self.pad[2],
                                             ignore_border=self.ignore_border,
                                             )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        pooled = dnn.dnn_pool(input, self.pool_size, self.stride,
                              self.mode, self.pad)
        return pooled


class MaxPool3DDNNLayer(Pool3DDNNLayer):
    """
    3D max-pooling layer

    Performs 3D max-pooling over the three trailing axes of a 5D input tensor.

    """
    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0, 0),
                 ignore_border=True, **kwargs):
        super(MaxPool3DDNNLayer, self).__init__(incoming, pool_size, stride,
                                                pad, ignore_border, mode='max',
                                                **kwargs)


def softmax_4dtensor(x):
    e_x = T.exp(x - x.max(axis=1, keepdims=True))
    e_x = e_x / e_x.sum(axis=1, keepdims=True)
    return e_x


def build_FCN(image_var, shape=(None, 1, None, None), n_class=1, load_vgg=False):
    # Build fully-connected network for semantic segmentation only
    net = {}

    net['in']      = L.InputLayer(shape, image_var)
    net['conv1_1'] = L.batch_norm(L.Conv2DLayer(net['in'], filter_size=3, num_filters=64, pad='same'))
    net['conv1_2'] = L.batch_norm(L.Conv2DLayer(net['conv1_1'], filter_size=3, num_filters=64, pad='same'))

    net['conv2_1'] = L.batch_norm(L.Conv2DLayer(net['conv1_2'], stride=2, filter_size=3, num_filters=128, pad='same'))
    net['conv2_2'] = L.batch_norm(L.Conv2DLayer(net['conv2_1'], filter_size=3, num_filters=128, pad='same'))

    net['conv3_1'] = L.batch_norm(L.Conv2DLayer(net['conv2_2'], stride=2, filter_size=3, num_filters=256, pad='same'))
    net['conv3_2'] = L.batch_norm(L.Conv2DLayer(net['conv3_1'], filter_size=3, num_filters=256, pad='same'))
    net['conv3_3'] = L.batch_norm(L.Conv2DLayer(net['conv3_2'], filter_size=3, num_filters=256, pad='same'))

    net['conv4_1'] = L.batch_norm(L.Conv2DLayer(net['conv3_3'], stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv4_2'] = L.batch_norm(L.Conv2DLayer(net['conv4_1'], filter_size=3, num_filters=512, pad='same'))
    net['conv4_3'] = L.batch_norm(L.Conv2DLayer(net['conv4_2'], filter_size=3, num_filters=512, pad='same'))

    net['conv5_1'] = L.batch_norm(L.Conv2DLayer(net['conv4_3'], stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv5_2'] = L.batch_norm(L.Conv2DLayer(net['conv5_1'], filter_size=3, num_filters=512, pad='same'))
    net['conv5_3'] = L.batch_norm(L.Conv2DLayer(net['conv5_2'], filter_size=3, num_filters=512, pad='same'))

    net['out1'] = L.batch_norm(L.Conv2DLayer(net['conv1_2'], filter_size=3, num_filters=64, pad='same'))
    net['out2'] = L.batch_norm(L.Conv2DLayer(net['conv2_2'], filter_size=3, num_filters=64, pad='same'))
    net['out3'] = L.batch_norm(L.Conv2DLayer(net['conv3_3'], filter_size=3, num_filters=64, pad='same'))
    net['out4'] = L.batch_norm(L.Conv2DLayer(net['conv4_3'], filter_size=3, num_filters=64, pad='same'))
    net['out5'] = L.batch_norm(L.Conv2DLayer(net['conv5_3'], filter_size=3, num_filters=64, pad='same'))

    net['out2_up'] = BilinearUpsamplingLayer(net['out2'], scale_factor=2)
    net['out3_up'] = BilinearUpsamplingLayer(net['out3'], scale_factor=4)
    net['out4_up'] = BilinearUpsamplingLayer(net['out4'], scale_factor=8)
    net['out5_up'] = BilinearUpsamplingLayer(net['out5'], scale_factor=16)

    net['concat'] = L.ConcatLayer([net['out1'],
                                   net['out2_up'],
                                   net['out3_up'],
                                   net['out4_up'],
                                   net['out5_up']])
    net['comb_1'] = L.Conv2DLayer(net['concat'], filter_size=1, num_filters=64, pad='same', nonlinearity=None)
    net['comb_2'] = L.batch_norm(L.Conv2DLayer(net['comb_1'], filter_size=1, num_filters=64, pad='same'))
    net['out']    = L.Conv2DLayer(net['comb_2'], filter_size=1, num_filters=n_class, pad='same', nonlinearity=softmax_4dtensor)

    # Initialise the weights for the combination layer so that concatenation is initially equivalent to summation
    print('Initialise the combination weights ...')
    W = np.zeros(net['comb_1'].get_params()[0].get_value().shape, dtype='float32')
    b = np.zeros(net['comb_1'].get_params()[1].get_value().shape, dtype='float32')
    for i in range(64):
        W[i, i::64] = 1.0
        b[i] = 0.0
    net['comb_1'].get_params()[0].set_value(W)
    net['comb_1'].get_params()[1].set_value(b)

    if load_vgg:
        # Initialise the convolutional layers using VGG16 weights
        print('Initialise the convolutional layers using VGG16 weights ...')

        with np.load('/vol/biomedic/users/wbai/data/deep_learning/VGG/VGG_ILSVRC_16_layers.npz') as f:
            vgg = f['vgg'][()]

            for layer_name in ['conv1_1', 'conv1_2',
                               'conv2_1', 'conv2_2',
                               'conv3_1', 'conv3_2', 'conv3_3',
                               'conv4_1', 'conv4_2', 'conv4_3',
                               'conv5_1', 'conv5_2', 'conv5_3']:
                # Since we apply batch_norm to the convolutional layer, each layer becomes Conv + BN + ReLU.
                # We need to find the original Conv layer by using .input_layer twice.
                # Also, batch_norm will remove the bias parameter b. Only W is kept.
                if layer_name == 'conv1_1':
                    W_mean = np.mean(vgg[layer_name]['W'], axis=1, keepdims=True)
                    net[layer_name].input_layer.input_layer.get_params()[0].set_value(np.repeat(W_mean, shape[1], axis=1))
                else:
                    net[layer_name].input_layer.input_layer.get_params()[0].set_value(vgg[layer_name]['W'])
    return net
    
def build_FCN_triple_branch(image_var, image_pred_var, image_seg_var, shape=(None, 1, None, None), n_class=1, load_vgg=False):
    # Build fully-connected network for motion estimation and semantic segmentation
    net = {}
    # Siamese-style motion estimation brach
    net['in']      = L.InputLayer(shape, image_var)
    net['conv1_1'] = L.batch_norm(L.Conv2DLayer(net['in'], filter_size=3, num_filters=64, pad='same'))
    net['conv1_2'] = L.batch_norm(L.Conv2DLayer(net['conv1_1'], filter_size=3, num_filters=64, pad='same'))

    net['conv2_1'] = L.batch_norm(L.Conv2DLayer(net['conv1_2'], stride=2, filter_size=3, num_filters=128, pad='same'))
    net['conv2_2'] = L.batch_norm(L.Conv2DLayer(net['conv2_1'], filter_size=3, num_filters=128, pad='same'))

    net['conv3_1'] = L.batch_norm(L.Conv2DLayer(net['conv2_2'], stride=2, filter_size=3, num_filters=256, pad='same'))
    net['conv3_2'] = L.batch_norm(L.Conv2DLayer(net['conv3_1'], filter_size=3, num_filters=256, pad='same'))
    net['conv3_3'] = L.batch_norm(L.Conv2DLayer(net['conv3_2'], filter_size=3, num_filters=256, pad='same'))

    net['conv4_1'] = L.batch_norm(L.Conv2DLayer(net['conv3_3'], stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv4_2'] = L.batch_norm(L.Conv2DLayer(net['conv4_1'], filter_size=3, num_filters=512, pad='same'))
    net['conv4_3'] = L.batch_norm(L.Conv2DLayer(net['conv4_2'], filter_size=3, num_filters=512, pad='same'))

    net['conv5_1'] = L.batch_norm(L.Conv2DLayer(net['conv4_3'], stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv5_2'] = L.batch_norm(L.Conv2DLayer(net['conv5_1'], filter_size=3, num_filters=512, pad='same'))
    net['conv5_3'] = L.batch_norm(L.Conv2DLayer(net['conv5_2'], filter_size=3, num_filters=512, pad='same'))
    
    net['in_pred'] = L.InputLayer(shape, image_pred_var)
    net['conv1_1s'] = L.batch_norm(L.Conv2DLayer(net['in_pred'], W = net['conv1_1'].input_layer.input_layer.W, filter_size=3, num_filters=64, pad='same'))
    net['conv1_2s'] = L.batch_norm(L.Conv2DLayer(net['conv1_1s'], W = net['conv1_2'].input_layer.input_layer.W,filter_size=3, num_filters=64, pad='same'))

    net['conv2_1s'] = L.batch_norm(L.Conv2DLayer(net['conv1_2s'], W = net['conv2_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=128, pad='same'))
    net['conv2_2s'] = L.batch_norm(L.Conv2DLayer(net['conv2_1s'], W = net['conv2_2'].input_layer.input_layer.W,filter_size=3, num_filters=128, pad='same'))

    net['conv3_1s'] = L.batch_norm(L.Conv2DLayer(net['conv2_2s'], W = net['conv3_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=256, pad='same'))
    net['conv3_2s'] = L.batch_norm(L.Conv2DLayer(net['conv3_1s'], W = net['conv3_2'].input_layer.input_layer.W,filter_size=3, num_filters=256, pad='same'))
    net['conv3_3s'] = L.batch_norm(L.Conv2DLayer(net['conv3_2s'], W = net['conv3_3'].input_layer.input_layer.W,filter_size=3, num_filters=256, pad='same'))

    net['conv4_1s'] = L.batch_norm(L.Conv2DLayer(net['conv3_3s'], W = net['conv4_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv4_2s'] = L.batch_norm(L.Conv2DLayer(net['conv4_1s'], W = net['conv4_2'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))
    net['conv4_3s'] = L.batch_norm(L.Conv2DLayer(net['conv4_2s'], W = net['conv4_3'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))

    net['conv5_1s'] = L.batch_norm(L.Conv2DLayer(net['conv4_3s'], W = net['conv5_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv5_2s'] = L.batch_norm(L.Conv2DLayer(net['conv5_1s'], W = net['conv5_2'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))
    net['conv5_3s'] = L.batch_norm(L.Conv2DLayer(net['conv5_2s'], W = net['conv5_3'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))

    net['concat1'] = L.ConcatLayer([net['conv1_2'], net['conv1_2s']])   
    net['concat2'] = L.ConcatLayer([net['conv2_2'], net['conv2_2s']])   
    net['concat3'] = L.ConcatLayer([net['conv3_3'], net['conv3_3s']])   
    net['concat4'] = L.ConcatLayer([net['conv4_3'], net['conv4_3s']])   
    net['concat5'] = L.ConcatLayer([net['conv5_3'], net['conv5_3s']])   
    
    net['out1'] = L.batch_norm(L.Conv2DLayer(net['concat1'], filter_size=3, num_filters=64, pad='same'))
    net['out2'] = L.batch_norm(L.Conv2DLayer(net['concat2'], filter_size=3, num_filters=64, pad='same'))
    net['out3'] = L.batch_norm(L.Conv2DLayer(net['concat3'], filter_size=3, num_filters=64, pad='same'))
    net['out4'] = L.batch_norm(L.Conv2DLayer(net['concat4'], filter_size=3, num_filters=64, pad='same'))
    net['out5'] = L.batch_norm(L.Conv2DLayer(net['concat5'], filter_size=3, num_filters=64, pad='same'))
    

    net['out2_up'] = BilinearUpsamplingLayer(net['out2'], scale_factor=2)
    net['out3_up'] = BilinearUpsamplingLayer(net['out3'], scale_factor=4)
    net['out4_up'] = BilinearUpsamplingLayer(net['out4'], scale_factor=8)
    net['out5_up'] = BilinearUpsamplingLayer(net['out5'], scale_factor=16)
    


    net['concat'] = L.ConcatLayer([net['out1'],
                                   net['out2_up'],
                                   net['out3_up'],
                                   net['out4_up'],
                                   net['out5_up']])
    net['comb_1'] = L.Conv2DLayer(net['concat'], filter_size=1, num_filters=64, pad='same', nonlinearity=None)
    net['comb_2'] = L.batch_norm(L.Conv2DLayer(net['comb_1'], filter_size=1, num_filters=64, pad='same'))
    net['out']    = L.Conv2DLayer(net['comb_2'], filter_size=1, num_filters=2, pad='same', nonlinearity=lasagne.nonlinearities.tanh)
    # Spatial Transformation (source to target)
    net['fr_st'] = OFLayer(net['in'],net['out'], name='fr_st')
    
    # Segmentation branch
    net['in_seg']    = L.InputLayer(shape, image_seg_var)
    net['conv1_1ss'] = L.batch_norm(L.Conv2DLayer(net['in_seg'], W = net['conv1_1'].input_layer.input_layer.W, filter_size=3, num_filters=64, pad='same'))
    net['conv1_2ss'] = L.batch_norm(L.Conv2DLayer(net['conv1_1ss'], W = net['conv1_2'].input_layer.input_layer.W,filter_size=3, num_filters=64, pad='same'))

    net['conv2_1ss'] = L.batch_norm(L.Conv2DLayer(net['conv1_2ss'], W = net['conv2_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=128, pad='same'))
    net['conv2_2ss'] = L.batch_norm(L.Conv2DLayer(net['conv2_1ss'], W = net['conv2_2'].input_layer.input_layer.W,filter_size=3, num_filters=128, pad='same'))

    net['conv3_1ss'] = L.batch_norm(L.Conv2DLayer(net['conv2_2ss'], W = net['conv3_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=256, pad='same'))
    net['conv3_2ss'] = L.batch_norm(L.Conv2DLayer(net['conv3_1ss'], W = net['conv3_2'].input_layer.input_layer.W,filter_size=3, num_filters=256, pad='same'))
    net['conv3_3ss'] = L.batch_norm(L.Conv2DLayer(net['conv3_2ss'], W = net['conv3_3'].input_layer.input_layer.W,filter_size=3, num_filters=256, pad='same'))

    net['conv4_1ss'] = L.batch_norm(L.Conv2DLayer(net['conv3_3ss'], W = net['conv4_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv4_2ss'] = L.batch_norm(L.Conv2DLayer(net['conv4_1ss'], W = net['conv4_2'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))
    net['conv4_3ss'] = L.batch_norm(L.Conv2DLayer(net['conv4_2ss'], W = net['conv4_3'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))

    net['conv5_1ss'] = L.batch_norm(L.Conv2DLayer(net['conv4_3ss'], W = net['conv5_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv5_2ss'] = L.batch_norm(L.Conv2DLayer(net['conv5_1ss'], W = net['conv5_2'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))
    net['conv5_3ss'] = L.batch_norm(L.Conv2DLayer(net['conv5_2ss'], W = net['conv5_3'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))

    net['out1s'] = L.batch_norm(L.Conv2DLayer(net['conv1_2ss'], filter_size=3, num_filters=64, pad='same'))
    net['out2s'] = L.batch_norm(L.Conv2DLayer(net['conv2_2ss'], filter_size=3, num_filters=64, pad='same'))
    net['out3s'] = L.batch_norm(L.Conv2DLayer(net['conv3_3ss'], filter_size=3, num_filters=64, pad='same'))
    net['out4s'] = L.batch_norm(L.Conv2DLayer(net['conv4_3ss'], filter_size=3, num_filters=64, pad='same'))
    net['out5s'] = L.batch_norm(L.Conv2DLayer(net['conv5_3ss'], filter_size=3, num_filters=64, pad='same'))

    net['out2s_up'] = BilinearUpsamplingLayer(net['out2s'], scale_factor=2)
    net['out3s_up'] = BilinearUpsamplingLayer(net['out3s'], scale_factor=4)
    net['out4s_up'] = BilinearUpsamplingLayer(net['out4s'], scale_factor=8)
    net['out5s_up'] = BilinearUpsamplingLayer(net['out5s'], scale_factor=16)

    net['concats'] = L.ConcatLayer([net['out1s'],
                                   net['out2s_up'],
                                   net['out3s_up'],
                                   net['out4s_up'],
                                   net['out5s_up']])
    net['comb_1s'] = L.Conv2DLayer(net['concats'], filter_size=1, num_filters=64, pad='same', nonlinearity=None)
    net['comb_2s'] = L.batch_norm(L.Conv2DLayer(net['comb_1s'], filter_size=1, num_filters=64, pad='same'))
    net['outs']    = L.Conv2DLayer(net['comb_2s'], filter_size=1, num_filters=4, pad='same', nonlinearity=softmax_4dtensor)
    # warp source segmentation to target
    net['warped_outs'] = OFLayer(net['outs'],net['out'], name='fr_st')

    return net
    
def build_FCN_triple_branch_rnn(image_var, image_pred_var, image_seg_var, shape=(None, 1, None, None, None), shape_seg = (None, 1, None, None), n_class=1, load_vgg=False):
    # Build network for joint motion estimation and segmentation with RNN
    net = {}

    net['in']      = L.InputLayer(shape, image_var)
    net['in'] = L.DimshuffleLayer(net['in'],(0,4,1,2,3))
    
    shape = L.get_output_shape(net['in']) #shape=[batch_size, seq_size, num_channel, width, height]
    n_channel = shape[2]
    batchsize = shape[0]
    seqlen = shape[1]
    width = shape[3]
    height = shape[4]
    # Reshape sequence input to batches dimension for easy extracting features
    net['in'] = L.ReshapeLayer(net['in'], (-1, n_channel, width, height))
    

    net['conv1_1'] = L.batch_norm(L.Conv2DLayer(net['in'], filter_size=3, num_filters=64, pad='same'))
    net['conv1_2'] = L.batch_norm(L.Conv2DLayer(net['conv1_1'], filter_size=3, num_filters=64, pad='same'))

    net['conv2_1'] = L.batch_norm(L.Conv2DLayer(net['conv1_2'], stride=2, filter_size=3, num_filters=128, pad='same'))
    net['conv2_2'] = L.batch_norm(L.Conv2DLayer(net['conv2_1'], filter_size=3, num_filters=128, pad='same'))

    net['conv3_1'] = L.batch_norm(L.Conv2DLayer(net['conv2_2'], stride=2, filter_size=3, num_filters=256, pad='same'))
    net['conv3_2'] = L.batch_norm(L.Conv2DLayer(net['conv3_1'], filter_size=3, num_filters=256, pad='same'))
    net['conv3_3'] = L.batch_norm(L.Conv2DLayer(net['conv3_2'], filter_size=3, num_filters=256, pad='same'))

    net['conv4_1'] = L.batch_norm(L.Conv2DLayer(net['conv3_3'], stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv4_2'] = L.batch_norm(L.Conv2DLayer(net['conv4_1'], filter_size=3, num_filters=512, pad='same'))
    net['conv4_3'] = L.batch_norm(L.Conv2DLayer(net['conv4_2'], filter_size=3, num_filters=512, pad='same'))

    net['conv5_1'] = L.batch_norm(L.Conv2DLayer(net['conv4_3'], stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv5_2'] = L.batch_norm(L.Conv2DLayer(net['conv5_1'], filter_size=3, num_filters=512, pad='same'))
    net['conv5_3'] = L.batch_norm(L.Conv2DLayer(net['conv5_2'], filter_size=3, num_filters=512, pad='same'))
    
    # somehow redundant, can be improved for efficiency
    net['in_pred'] = L.InputLayer(shape, image_pred_var)
    net['in_pred'] = L.DimshuffleLayer(net['in_pred'],(0,4,1,2,3))
    net['in_pred'] = L.ReshapeLayer(net['in_pred'], (-1, n_channel, width, height))
    net['conv1_1s'] = L.batch_norm(L.Conv2DLayer(net['in_pred'], W = net['conv1_1'].input_layer.input_layer.W, filter_size=3, num_filters=64, pad='same'))
    net['conv1_2s'] = L.batch_norm(L.Conv2DLayer(net['conv1_1s'], W = net['conv1_2'].input_layer.input_layer.W,filter_size=3, num_filters=64, pad='same'))

    net['conv2_1s'] = L.batch_norm(L.Conv2DLayer(net['conv1_2s'], W = net['conv2_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=128, pad='same'))
    net['conv2_2s'] = L.batch_norm(L.Conv2DLayer(net['conv2_1s'], W = net['conv2_2'].input_layer.input_layer.W,filter_size=3, num_filters=128, pad='same'))

    net['conv3_1s'] = L.batch_norm(L.Conv2DLayer(net['conv2_2s'], W = net['conv3_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=256, pad='same'))
    net['conv3_2s'] = L.batch_norm(L.Conv2DLayer(net['conv3_1s'], W = net['conv3_2'].input_layer.input_layer.W,filter_size=3, num_filters=256, pad='same'))
    net['conv3_3s'] = L.batch_norm(L.Conv2DLayer(net['conv3_2s'], W = net['conv3_3'].input_layer.input_layer.W,filter_size=3, num_filters=256, pad='same'))

    net['conv4_1s'] = L.batch_norm(L.Conv2DLayer(net['conv3_3s'], W = net['conv4_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv4_2s'] = L.batch_norm(L.Conv2DLayer(net['conv4_1s'], W = net['conv4_2'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))
    net['conv4_3s'] = L.batch_norm(L.Conv2DLayer(net['conv4_2s'], W = net['conv4_3'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))

    net['conv5_1s'] = L.batch_norm(L.Conv2DLayer(net['conv4_3s'], W = net['conv5_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv5_2s'] = L.batch_norm(L.Conv2DLayer(net['conv5_1s'], W = net['conv5_2'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))
    net['conv5_3s'] = L.batch_norm(L.Conv2DLayer(net['conv5_2s'], W = net['conv5_3'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))

    net['concat1'] = L.ConcatLayer([net['conv1_2'], net['conv1_2s']])   
    net['concat2'] = L.ConcatLayer([net['conv2_2'], net['conv2_2s']])   
    net['concat3'] = L.ConcatLayer([net['conv3_3'], net['conv3_3s']])   
    net['concat4'] = L.ConcatLayer([net['conv4_3'], net['conv4_3s']])   
    net['concat5'] = L.ConcatLayer([net['conv5_3'], net['conv5_3s']])   
    
    net['out1'] = L.batch_norm(L.Conv2DLayer(net['concat1'], filter_size=3, num_filters=64, pad='same'))
    net['out2'] = L.batch_norm(L.Conv2DLayer(net['concat2'], filter_size=3, num_filters=64, pad='same'))
    net['out3'] = L.batch_norm(L.Conv2DLayer(net['concat3'], filter_size=3, num_filters=64, pad='same'))
    net['out4'] = L.batch_norm(L.Conv2DLayer(net['concat4'], filter_size=3, num_filters=64, pad='same'))
    net['out5'] = L.batch_norm(L.Conv2DLayer(net['concat5'], filter_size=3, num_filters=64, pad='same'))
    

    net['out2_up'] = BilinearUpsamplingLayer(net['out2'], scale_factor=2)
    net['out3_up'] = BilinearUpsamplingLayer(net['out3'], scale_factor=4)
    net['out4_up'] = BilinearUpsamplingLayer(net['out4'], scale_factor=8)
    net['out5_up'] = BilinearUpsamplingLayer(net['out5'], scale_factor=16)
    


    net['concat'] = L.ConcatLayer([net['out1'],
                                   net['out2_up'],
                                   net['out3_up'],
                                   net['out4_up'],
                                   net['out5_up']])
    net['comb_1'] = L.Conv2DLayer(net['concat'], filter_size=1, num_filters=64, pad='same', nonlinearity=None)
    net['comb_2'] = L.batch_norm(L.Conv2DLayer(net['comb_1'], filter_size=1, num_filters=64, pad='same'))
    net['comb_2_rshp'] =  L.ReshapeLayer(net['comb_2'],(-1, seqlen, 64, width, height))
    
    net['in_to_hid'] = L.Conv2DLayer(L.InputLayer((None, 64, width, height)), num_filters=2, filter_size=1, nonlinearity=None, name ='in_to_hid' )
    net['hid_to_hid'] = L.Conv2DLayer(L.InputLayer(net['in_to_hid'].output_shape), num_filters=2, filter_size=1, nonlinearity=None, name = 'hid_to_hid')
    net['rec'] = L.CustomRecurrentLayer(net['comb_2_rshp'], net['in_to_hid'], net['hid_to_hid'],nonlinearity=lasagne.nonlinearities.tanh, name = 'rec')
	    
    net['out'] = L.ReshapeLayer(net['rec'], (-1, 2, width, height))     
    
    #net['out']    = L.Conv2DLayer(net['comb_2'], filter_size=1, num_filters=2, pad='same', nonlinearity=lasagne.nonlinearities.tanh)
    net['fr_st'] = OFLayer(net['in'],net['out'], name='fr_st')
    net['fr_st'] =  L.ReshapeLayer(net['fr_st'],(-1, seqlen, n_channel, width, height))
    net['fr_st'] = L.DimshuffleLayer(net['fr_st'],(0,2,3,4,1))
    
    net['in_seg']    = L.InputLayer(shape_seg, image_seg_var)
#    net['in_seg'] = L.DimshuffleLayer(net['in_seg'],(0,4,1,2,3))
#    net['in_seg'] = L.ReshapeLayer(net['in_seg'], (-1, n_channel, width, height))
    net['conv1_1ss'] = L.batch_norm(L.Conv2DLayer(net['in_seg'], W = net['conv1_1'].input_layer.input_layer.W, filter_size=3, num_filters=64, pad='same'))
    net['conv1_2ss'] = L.batch_norm(L.Conv2DLayer(net['conv1_1ss'], W = net['conv1_2'].input_layer.input_layer.W,filter_size=3, num_filters=64, pad='same'))

    net['conv2_1ss'] = L.batch_norm(L.Conv2DLayer(net['conv1_2ss'], W = net['conv2_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=128, pad='same'))
    net['conv2_2ss'] = L.batch_norm(L.Conv2DLayer(net['conv2_1ss'], W = net['conv2_2'].input_layer.input_layer.W,filter_size=3, num_filters=128, pad='same'))

    net['conv3_1ss'] = L.batch_norm(L.Conv2DLayer(net['conv2_2ss'], W = net['conv3_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=256, pad='same'))
    net['conv3_2ss'] = L.batch_norm(L.Conv2DLayer(net['conv3_1ss'], W = net['conv3_2'].input_layer.input_layer.W,filter_size=3, num_filters=256, pad='same'))
    net['conv3_3ss'] = L.batch_norm(L.Conv2DLayer(net['conv3_2ss'], W = net['conv3_3'].input_layer.input_layer.W,filter_size=3, num_filters=256, pad='same'))

    net['conv4_1ss'] = L.batch_norm(L.Conv2DLayer(net['conv3_3ss'], W = net['conv4_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv4_2ss'] = L.batch_norm(L.Conv2DLayer(net['conv4_1ss'], W = net['conv4_2'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))
    net['conv4_3ss'] = L.batch_norm(L.Conv2DLayer(net['conv4_2ss'], W = net['conv4_3'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))

    net['conv5_1ss'] = L.batch_norm(L.Conv2DLayer(net['conv4_3ss'], W = net['conv5_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv5_2ss'] = L.batch_norm(L.Conv2DLayer(net['conv5_1ss'], W = net['conv5_2'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))
    net['conv5_3ss'] = L.batch_norm(L.Conv2DLayer(net['conv5_2ss'], W = net['conv5_3'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))

    net['out1s'] = L.batch_norm(L.Conv2DLayer(net['conv1_2ss'], filter_size=3, num_filters=64, pad='same'))
    net['out2s'] = L.batch_norm(L.Conv2DLayer(net['conv2_2ss'], filter_size=3, num_filters=64, pad='same'))
    net['out3s'] = L.batch_norm(L.Conv2DLayer(net['conv3_3ss'], filter_size=3, num_filters=64, pad='same'))
    net['out4s'] = L.batch_norm(L.Conv2DLayer(net['conv4_3ss'], filter_size=3, num_filters=64, pad='same'))
    net['out5s'] = L.batch_norm(L.Conv2DLayer(net['conv5_3ss'], filter_size=3, num_filters=64, pad='same'))

    net['out2s_up'] = BilinearUpsamplingLayer(net['out2s'], scale_factor=2)
    net['out3s_up'] = BilinearUpsamplingLayer(net['out3s'], scale_factor=4)
    net['out4s_up'] = BilinearUpsamplingLayer(net['out4s'], scale_factor=8)
    net['out5s_up'] = BilinearUpsamplingLayer(net['out5s'], scale_factor=16)

    net['concats'] = L.ConcatLayer([net['out1s'],
                                   net['out2s_up'],
                                   net['out3s_up'],
                                   net['out4s_up'],
                                   net['out5s_up']])
    net['comb_1s'] = L.Conv2DLayer(net['concats'], filter_size=1, num_filters=64, pad='same', nonlinearity=None)
    net['comb_2s'] = L.batch_norm(L.Conv2DLayer(net['comb_1s'], filter_size=1, num_filters=64, pad='same'))
    net['outs']    = L.Conv2DLayer(net['comb_2s'], filter_size=1, num_filters=4, pad='same', nonlinearity=softmax_4dtensor)
    net['warped_outs'] = OFLayer(net['outs'],net['out'], name='fr_st')
    #net['warped_outs'] =  L.ReshapeLayer(net['warped_outs'],(-1, seqlen, n_channel, width, height))
    #net['warped_outs'] = L.DimshuffleLayer(net['warped_outs'],(0,2,3,4,1))
    return net
    
def build_FCN_Siemese_flow(image_var, image_pred_var, shape=(None, 1, None, None), n_class=1, load_vgg=False):
    # Build fully-connected network for motion estimation only
    net = {}

    net['in']      = L.InputLayer(shape, image_var)
    net['conv1_1'] = L.batch_norm(L.Conv2DLayer(net['in'], filter_size=3, num_filters=64, pad='same'))
    net['conv1_2'] = L.batch_norm(L.Conv2DLayer(net['conv1_1'], filter_size=3, num_filters=64, pad='same'))

    net['conv2_1'] = L.batch_norm(L.Conv2DLayer(net['conv1_2'], stride=2, filter_size=3, num_filters=128, pad='same'))
    net['conv2_2'] = L.batch_norm(L.Conv2DLayer(net['conv2_1'], filter_size=3, num_filters=128, pad='same'))

    net['conv3_1'] = L.batch_norm(L.Conv2DLayer(net['conv2_2'], stride=2, filter_size=3, num_filters=256, pad='same'))
    net['conv3_2'] = L.batch_norm(L.Conv2DLayer(net['conv3_1'], filter_size=3, num_filters=256, pad='same'))
    net['conv3_3'] = L.batch_norm(L.Conv2DLayer(net['conv3_2'], filter_size=3, num_filters=256, pad='same'))

    net['conv4_1'] = L.batch_norm(L.Conv2DLayer(net['conv3_3'], stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv4_2'] = L.batch_norm(L.Conv2DLayer(net['conv4_1'], filter_size=3, num_filters=512, pad='same'))
    net['conv4_3'] = L.batch_norm(L.Conv2DLayer(net['conv4_2'], filter_size=3, num_filters=512, pad='same'))

    net['conv5_1'] = L.batch_norm(L.Conv2DLayer(net['conv4_3'], stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv5_2'] = L.batch_norm(L.Conv2DLayer(net['conv5_1'], filter_size=3, num_filters=512, pad='same'))
    net['conv5_3'] = L.batch_norm(L.Conv2DLayer(net['conv5_2'], filter_size=3, num_filters=512, pad='same'))
    
    net['in_pred'] = L.InputLayer(shape, image_pred_var)
    net['conv1_1s'] = L.batch_norm(L.Conv2DLayer(net['in_pred'], W = net['conv1_1'].input_layer.input_layer.W, filter_size=3, num_filters=64, pad='same'))
    net['conv1_2s'] = L.batch_norm(L.Conv2DLayer(net['conv1_1s'], W = net['conv1_2'].input_layer.input_layer.W,filter_size=3, num_filters=64, pad='same'))

    net['conv2_1s'] = L.batch_norm(L.Conv2DLayer(net['conv1_2s'], W = net['conv2_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=128, pad='same'))
    net['conv2_2s'] = L.batch_norm(L.Conv2DLayer(net['conv2_1s'], W = net['conv2_2'].input_layer.input_layer.W,filter_size=3, num_filters=128, pad='same'))

    net['conv3_1s'] = L.batch_norm(L.Conv2DLayer(net['conv2_2s'], W = net['conv3_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=256, pad='same'))
    net['conv3_2s'] = L.batch_norm(L.Conv2DLayer(net['conv3_1s'], W = net['conv3_2'].input_layer.input_layer.W,filter_size=3, num_filters=256, pad='same'))
    net['conv3_3s'] = L.batch_norm(L.Conv2DLayer(net['conv3_2s'], W = net['conv3_3'].input_layer.input_layer.W,filter_size=3, num_filters=256, pad='same'))

    net['conv4_1s'] = L.batch_norm(L.Conv2DLayer(net['conv3_3s'], W = net['conv4_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv4_2s'] = L.batch_norm(L.Conv2DLayer(net['conv4_1s'], W = net['conv4_2'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))
    net['conv4_3s'] = L.batch_norm(L.Conv2DLayer(net['conv4_2s'], W = net['conv4_3'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))

    net['conv5_1s'] = L.batch_norm(L.Conv2DLayer(net['conv4_3s'], W = net['conv5_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv5_2s'] = L.batch_norm(L.Conv2DLayer(net['conv5_1s'], W = net['conv5_2'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))
    net['conv5_3s'] = L.batch_norm(L.Conv2DLayer(net['conv5_2s'], W = net['conv5_3'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))

    net['concat1'] = L.ConcatLayer([net['conv1_2'], net['conv1_2s']])   
    net['concat2'] = L.ConcatLayer([net['conv2_2'], net['conv2_2s']])   
    net['concat3'] = L.ConcatLayer([net['conv3_3'], net['conv3_3s']])   
    net['concat4'] = L.ConcatLayer([net['conv4_3'], net['conv4_3s']])   
    net['concat5'] = L.ConcatLayer([net['conv5_3'], net['conv5_3s']])   
    
    net['out1'] = L.batch_norm(L.Conv2DLayer(net['concat1'], filter_size=3, num_filters=64, pad='same'))
    net['out2'] = L.batch_norm(L.Conv2DLayer(net['concat2'], filter_size=3, num_filters=64, pad='same'))
    net['out3'] = L.batch_norm(L.Conv2DLayer(net['concat3'], filter_size=3, num_filters=64, pad='same'))
    net['out4'] = L.batch_norm(L.Conv2DLayer(net['concat4'], filter_size=3, num_filters=64, pad='same'))
    net['out5'] = L.batch_norm(L.Conv2DLayer(net['concat5'], filter_size=3, num_filters=64, pad='same'))
    

    net['out2_up'] = BilinearUpsamplingLayer(net['out2'], scale_factor=2)
    net['out3_up'] = BilinearUpsamplingLayer(net['out3'], scale_factor=4)
    net['out4_up'] = BilinearUpsamplingLayer(net['out4'], scale_factor=8)
    net['out5_up'] = BilinearUpsamplingLayer(net['out5'], scale_factor=16)
    


    net['concat'] = L.ConcatLayer([net['out1'],
                                   net['out2_up'],
                                   net['out3_up'],
                                   net['out4_up'],
                                   net['out5_up']])
    net['comb_1'] = L.Conv2DLayer(net['concat'], filter_size=1, num_filters=64, pad='same', nonlinearity=None)
    net['comb_2'] = L.batch_norm(L.Conv2DLayer(net['comb_1'], filter_size=1, num_filters=64, pad='same'))
    net['out']    = L.Conv2DLayer(net['comb_2'], filter_size=1, num_filters=2, pad='same', nonlinearity=lasagne.nonlinearities.tanh)
    net['fr_st'] = OFLayer(net['in'],net['out'], name='fr_st')
    # Initialise the weights for the combination layer so that concatenation is initially equivalent to summation
    print('Initialise the combination weights ...')
    W = np.zeros(net['comb_1'].get_params()[0].get_value().shape, dtype='float32')
    b = np.zeros(net['comb_1'].get_params()[1].get_value().shape, dtype='float32')
    for i in range(64):
        W[i, i::64] = 1.0
        b[i] = 0.0
    net['comb_1'].get_params()[0].set_value(W)
    net['comb_1'].get_params()[1].set_value(b)

    if load_vgg:
        # Initialise the convolutional layers using VGG16 weights
        print('Initialise the convolutional layers using VGG16 weights ...')

        with np.load('/vol/biomedic/users/wbai/data/deep_learning/VGG/VGG_ILSVRC_16_layers.npz') as f:
            vgg = f['vgg'][()]

            for layer_name in ['conv1_1', 'conv1_2',
                               'conv2_1', 'conv2_2',
                               'conv3_1', 'conv3_2', 'conv3_3',
                               'conv4_1', 'conv4_2', 'conv4_3',
                               'conv5_1', 'conv5_2', 'conv5_3']:
                # Since we apply batch_norm to the convolutional layer, each layer becomes Conv + BN + ReLU.
                # We need to find the original Conv layer by using .input_layer twice.
                # Also, batch_norm will remove the bias parameter b. Only W is kept.
                if layer_name == 'conv1_1':
                    W_mean = np.mean(vgg[layer_name]['W'], axis=1, keepdims=True)
                    net[layer_name].input_layer.input_layer.get_params()[0].set_value(np.repeat(W_mean, shape[1], axis=1))
                else:
                    net[layer_name].input_layer.input_layer.get_params()[0].set_value(vgg[layer_name]['W'])
    return net
    
def build_FCN_Siemese_flow_rnn(image_var, image_pred_var, shape=(None, 1, None, None, None), n_class=1, load_vgg=False):
    # Build fully-connected network for motion estimation only with RNN component
    net = {}

    net['in']      = L.InputLayer(shape, image_var)
    net['in'] = L.DimshuffleLayer(net['in'],(0,4,1,2,3))
    
    shape = L.get_output_shape(net['in']) #shape=[batch_size, seq_size, num_channel, width, height]
    n_channel = shape[2]
    batchsize = shape[0]
    seqlen = shape[1]
    width = shape[3]
    height = shape[4]
    
    net['in'] = L.ReshapeLayer(net['in'], (-1, n_channel, width, height))
    

    net['conv1_1'] = L.batch_norm(L.Conv2DLayer(net['in'], filter_size=3, num_filters=64, pad='same'))
    net['conv1_2'] = L.batch_norm(L.Conv2DLayer(net['conv1_1'], filter_size=3, num_filters=64, pad='same'))

    net['conv2_1'] = L.batch_norm(L.Conv2DLayer(net['conv1_2'], stride=2, filter_size=3, num_filters=128, pad='same'))
    net['conv2_2'] = L.batch_norm(L.Conv2DLayer(net['conv2_1'], filter_size=3, num_filters=128, pad='same'))

    net['conv3_1'] = L.batch_norm(L.Conv2DLayer(net['conv2_2'], stride=2, filter_size=3, num_filters=256, pad='same'))
    net['conv3_2'] = L.batch_norm(L.Conv2DLayer(net['conv3_1'], filter_size=3, num_filters=256, pad='same'))
    net['conv3_3'] = L.batch_norm(L.Conv2DLayer(net['conv3_2'], filter_size=3, num_filters=256, pad='same'))

    net['conv4_1'] = L.batch_norm(L.Conv2DLayer(net['conv3_3'], stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv4_2'] = L.batch_norm(L.Conv2DLayer(net['conv4_1'], filter_size=3, num_filters=512, pad='same'))
    net['conv4_3'] = L.batch_norm(L.Conv2DLayer(net['conv4_2'], filter_size=3, num_filters=512, pad='same'))

    net['conv5_1'] = L.batch_norm(L.Conv2DLayer(net['conv4_3'], stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv5_2'] = L.batch_norm(L.Conv2DLayer(net['conv5_1'], filter_size=3, num_filters=512, pad='same'))
    net['conv5_3'] = L.batch_norm(L.Conv2DLayer(net['conv5_2'], filter_size=3, num_filters=512, pad='same'))
    
    net['in_pred'] = L.InputLayer(shape, image_pred_var)
    net['in_pred'] = L.DimshuffleLayer(net['in_pred'],(0,4,1,2,3))
    net['in_pred'] = L.ReshapeLayer(net['in_pred'], (-1, n_channel, width, height))
    net['conv1_1s'] = L.batch_norm(L.Conv2DLayer(net['in_pred'], W = net['conv1_1'].input_layer.input_layer.W, filter_size=3, num_filters=64, pad='same'))
    net['conv1_2s'] = L.batch_norm(L.Conv2DLayer(net['conv1_1s'], W = net['conv1_2'].input_layer.input_layer.W,filter_size=3, num_filters=64, pad='same'))

    net['conv2_1s'] = L.batch_norm(L.Conv2DLayer(net['conv1_2s'], W = net['conv2_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=128, pad='same'))
    net['conv2_2s'] = L.batch_norm(L.Conv2DLayer(net['conv2_1s'], W = net['conv2_2'].input_layer.input_layer.W,filter_size=3, num_filters=128, pad='same'))

    net['conv3_1s'] = L.batch_norm(L.Conv2DLayer(net['conv2_2s'], W = net['conv3_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=256, pad='same'))
    net['conv3_2s'] = L.batch_norm(L.Conv2DLayer(net['conv3_1s'], W = net['conv3_2'].input_layer.input_layer.W,filter_size=3, num_filters=256, pad='same'))
    net['conv3_3s'] = L.batch_norm(L.Conv2DLayer(net['conv3_2s'], W = net['conv3_3'].input_layer.input_layer.W,filter_size=3, num_filters=256, pad='same'))

    net['conv4_1s'] = L.batch_norm(L.Conv2DLayer(net['conv3_3s'], W = net['conv4_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv4_2s'] = L.batch_norm(L.Conv2DLayer(net['conv4_1s'], W = net['conv4_2'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))
    net['conv4_3s'] = L.batch_norm(L.Conv2DLayer(net['conv4_2s'], W = net['conv4_3'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))

    net['conv5_1s'] = L.batch_norm(L.Conv2DLayer(net['conv4_3s'], W = net['conv5_1'].input_layer.input_layer.W,stride=2, filter_size=3, num_filters=512, pad='same'))
    net['conv5_2s'] = L.batch_norm(L.Conv2DLayer(net['conv5_1s'], W = net['conv5_2'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))
    net['conv5_3s'] = L.batch_norm(L.Conv2DLayer(net['conv5_2s'], W = net['conv5_3'].input_layer.input_layer.W,filter_size=3, num_filters=512, pad='same'))

    net['concat1'] = L.ConcatLayer([net['conv1_2'], net['conv1_2s']])   
    net['concat2'] = L.ConcatLayer([net['conv2_2'], net['conv2_2s']])   
    net['concat3'] = L.ConcatLayer([net['conv3_3'], net['conv3_3s']])   
    net['concat4'] = L.ConcatLayer([net['conv4_3'], net['conv4_3s']])   
    net['concat5'] = L.ConcatLayer([net['conv5_3'], net['conv5_3s']])   
    
    net['out1'] = L.batch_norm(L.Conv2DLayer(net['concat1'], filter_size=3, num_filters=64, pad='same'))
    net['out2'] = L.batch_norm(L.Conv2DLayer(net['concat2'], filter_size=3, num_filters=64, pad='same'))
    net['out3'] = L.batch_norm(L.Conv2DLayer(net['concat3'], filter_size=3, num_filters=64, pad='same'))
    net['out4'] = L.batch_norm(L.Conv2DLayer(net['concat4'], filter_size=3, num_filters=64, pad='same'))
    net['out5'] = L.batch_norm(L.Conv2DLayer(net['concat5'], filter_size=3, num_filters=64, pad='same'))
    

    net['out2_up'] = BilinearUpsamplingLayer(net['out2'], scale_factor=2)
    net['out3_up'] = BilinearUpsamplingLayer(net['out3'], scale_factor=4)
    net['out4_up'] = BilinearUpsamplingLayer(net['out4'], scale_factor=8)
    net['out5_up'] = BilinearUpsamplingLayer(net['out5'], scale_factor=16)
    


    net['concat'] = L.ConcatLayer([net['out1'],
                                   net['out2_up'],
                                   net['out3_up'],
                                   net['out4_up'],
                                   net['out5_up']])
    net['comb_1'] = L.Conv2DLayer(net['concat'], filter_size=1, num_filters=64, pad='same', nonlinearity=None)
    net['comb_2'] = L.batch_norm(L.Conv2DLayer(net['comb_1'], filter_size=1, num_filters=64, pad='same'))
    net['comb_2_rshp'] =  L.ReshapeLayer(net['comb_2'],(-1, seqlen, 64, width, height))
    
    net['in_to_hid'] = L.Conv2DLayer(L.InputLayer((None, 64, width, height)), num_filters=2, filter_size=1, nonlinearity=None, name ='in_to_hid' )
    net['hid_to_hid'] = L.Conv2DLayer(L.InputLayer(net['in_to_hid'].output_shape), num_filters=2, filter_size=1, nonlinearity=None, name = 'hid_to_hid')
    net['rec'] = L.CustomRecurrentLayer(net['comb_2_rshp'], net['in_to_hid'], net['hid_to_hid'],nonlinearity=lasagne.nonlinearities.tanh, name = 'rec')
	    
    net['out'] = L.ReshapeLayer(net['rec'], (-1, 2, width, height))     
    
    #net['out']    = L.Conv2DLayer(net['comb_2'], filter_size=1, num_filters=2, pad='same', nonlinearity=lasagne.nonlinearities.tanh)
    net['fr_st'] = OFLayer(net['in'],net['out'], name='fr_st')
    net['fr_st'] =  L.ReshapeLayer(net['fr_st'],(-1, seqlen, n_channel, width, height))
    net['fr_st'] = L.DimshuffleLayer(net['fr_st'],(0,2,3,4,1))

    return net
    
