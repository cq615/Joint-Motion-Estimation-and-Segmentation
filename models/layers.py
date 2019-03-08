import theano.tensor as T
import lasagne
from lasagne.layers import Layer

import theano
import numpy as np
from lasagne.layers import MergeLayer, Conv2DLayer

def ensure_set_name(default_name, kwargs):
    """Ensure that the parameters contain names. Be careful, kwargs need to be
    passed as a dictionary here

    Parameters
    ----------
    default_name: string
        default name to set if neither name or pr is present, or if name is not
        present but pr is, the name becomes ``pr+default_name''
    kwargs: dict
        keyword arguments given to functions

    Returns
    -------
    kwargs: dict
    """
    if 'name' not in kwargs:
        raise Warning("You need to name the layers, "
                      "otherwise it simply won't work")
    global id_ctr
    if 'name' in kwargs and 'pr' in kwargs:
        kwargs['name'] = kwargs['pr']+kwargs['name']
    elif 'name' not in kwargs and 'pr' in kwargs:
        idx = next(id_ctr)
        kwargs['name'] = kwargs['pr'] + default_name + '_g' + str(idx)
    elif 'name' not in kwargs:
        idx = next(id_ctr)
        kwargs['name'] = default_name + '_g' + str(idx)
    return kwargs

def Conv(incoming, num_filters, filter_size=3,
         stride=(1, 1), pad='same', W=lasagne.init.HeNormal(),
         b=None, nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
    """
    Overrides the default parameters for ConvLayer
    """
    ensure_set_name('conv', kwargs)

    return Conv2DLayer(incoming, num_filters, filter_size, stride, pad, W=W, b=b,
                     nonlinearity=nonlinearity, **kwargs)

def as_tuple(x, N, t=None):
    """
    Coerce a value to a tuple of given length (and possibly given type).

    Parameters
    ----------
    x : value or iterable
    N : integer
        length of the desired tuple
    t : type, optional
        required type for all elements

    Returns
    -------
    tuple
        ``tuple(x)`` if `x` is iterable, ``(x,) * N`` otherwise.

    Raises
    ------
    TypeError
        if `type` is given and `x` or any of its elements do not match it
    ValueError
        if `x` is iterable, but does not have exactly `N` elements
    """
    try:
        X = tuple(x)
    except TypeError:
        X = (x,) * N

    if (t is not None) and not all(isinstance(v, t) for v in X):
        raise TypeError("expected a single value or an iterable "
                        "of {0}, got {1} instead".format(t.__name__, x))

    if len(X) != N:
        raise ValueError("expected a single value or an iterable "
                         "with length {0}, got {1} instead".format(N, x))

    return X


class IdLayer(Layer):
    def get_output_for(self, input, **kwargs):
        return input


class SumLayer(Layer):
    def get_output_for(self, input, **kwargs):
        return input.sum(axis=-1)

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1]


class SHLULayer(Layer):
    def get_output_for(self, input, **kwargs):
        return T.sgn(input) * T.maximum(input - 1, 0)


class ResidualLayer(lasagne.layers.ElemwiseSumLayer):
    '''
    Residual Layer, which just wraps around ElemwiseSumLayer
    '''

    def __init__(self, incomings, **kwargs):
        ensure_set_name('res', kwargs)
        super(ResidualLayer, self).__init__(incomings, **kwargs)
        # store names
        input_names = []
        for l in incomings:
            if isinstance(l, lasagne.layers.InputLayer):
                input_names.append(l.name if l.name else l.input_var.name)
            elif l.name:
                input_names.append(l.name)
            else:
                input_names.append(str(l))

        self.input_names = input_names

    def get_output_for(self, inputs, **kwargs):
        return super(lasagne.layers.ElemwiseSumLayer,
                     self).get_output_for(inputs, **kwargs)

class OFLayer(MergeLayer):
    def __init__(self, incoming, localization_network, downsample_factor=1,
                 border_mode='nearest', **kwargs):
        super(OFLayer, self).__init__(
            [incoming, localization_network], **kwargs)
        self.downsample_factor = as_tuple(downsample_factor, 2)
        self.border_mode = border_mode

        input_shp, loc_shp = self.input_shapes
#        loc_shp=(batch_size,2,height, width)
#        if loc_shp[-1] != 6 or len(loc_shp) != 2:
#            raise ValueError("The localization network must have "
#                             "output shape: (batch_size, 6)")
        if len(input_shp) != 4:
            raise ValueError("The input network must have a 4-dimensional "
                             "output shape: (batch_size, num_input_channels, "
                             "input_rows, input_columns)")

    def get_output_shape_for(self, input_shapes):
        shape = input_shapes[0]
        factors = self.downsample_factor
        return (shape[:2] + tuple(None if s is None else int(s // f)
                                  for s, f in zip(shape[2:], factors)))

    def get_output_for(self, inputs, **kwargs):
        # see eq. (1) and sec 3.1 in [1]
        input, theta = inputs
        return _transform(theta, input, self.downsample_factor,
                                 self.border_mode)


def _transform(theta, input, downsample_factor, border_mode):
    num_batch, num_channels, height, width = input.shape
#    theta = T.reshape(theta, (-1, 2, 3))

    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    out_height = T.cast(height // downsample_factor[0], 'int32')
    out_width = T.cast(width // downsample_factor[1], 'int32')
    grid = _meshgrid(out_height, out_width)

    # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
    
#    T_g = T.dot(theta, grid)
    T_d = T.reshape(theta, (-1,2,out_height*out_width))
    grid = grid.reshape((1,grid.shape[0], grid.shape[1])).repeat(T_d.shape[0],0)
    T_g = T_d+grid
    #T_g = (T_d+1) * grid;
    #T_g = T_d[:,2:,:] + T_g;
    x_s = T_g[:, 0]
    y_s = T_g[:, 1]
    x_s_flat = x_s.flatten()
    y_s_flat = y_s.flatten()

    # dimshuffle input to  (bs, height, width, channels)
    input_dim = input.dimshuffle(0, 2, 3, 1)
    input_transformed = _interpolate(
        input_dim, x_s_flat, y_s_flat,
        out_height, out_width, border_mode)

    output = T.reshape(
        input_transformed, (num_batch, out_height, out_width, num_channels))
    output = output.dimshuffle(0, 3, 1, 2)  # dimshuffle to conv format
    return output
    
    


def _interpolate(im, x, y, out_height, out_width, border_mode):
    # *_f are floats
    num_batch, height, width, channels = im.shape
    height_f = T.cast(height, theano.config.floatX)
    width_f = T.cast(width, theano.config.floatX)

    # scale coordinates from [-1, 1] to [0, width/height - 1]
    x = (x + 1) / 2 * (width_f - 1)
    y = (y + 1) / 2 * (height_f - 1)

    # obtain indices of the 2x2 pixel neighborhood surrounding the coordinates;
    # we need those in floatX for interpolation and in int64 for indexing.
    x0_f = T.floor(x)
    y0_f = T.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1

    # for indexing, we need to take care of the border mode for outside pixels.
    if border_mode == 'nearest':
        x0 = T.clip(x0_f, 0, width_f - 1)
        x1 = T.clip(x1_f, 0, width_f - 1)
        y0 = T.clip(y0_f, 0, height_f - 1)
        y1 = T.clip(y1_f, 0, height_f - 1)
    elif border_mode == 'mirror':
        w = 2 * (width_f - 1)
        x0 = T.minimum(x0_f % w, -x0_f % w)
        x1 = T.minimum(x1_f % w, -x1_f % w)
        h = 2 * (height_f - 1)
        y0 = T.minimum(y0_f % h, -y0_f % h)
        y1 = T.minimum(y1_f % h, -y1_f % h)
    elif border_mode == 'wrap':
        x0 = T.mod(x0_f, width_f)
        x1 = T.mod(x1_f, width_f)
        y0 = T.mod(y0_f, height_f)
        y1 = T.mod(y1_f, height_f)
    else:
        raise ValueError("border_mode must be one of "
                         "'nearest', 'mirror', 'wrap'")
    x0, x1, y0, y1 = (T.cast(v, 'int32') for v in (x0, x1, y0, y1))

    # The input is [num_batch, height, width, channels]. We do the lookup in
    # the flattened input, i.e [num_batch*height*width, channels]. We need
    # to offset all indices to match the flat version
    dim2 = width
    dim1 = width*height
    base = T.repeat(
        T.arange(num_batch, dtype='int32')*dim1, out_height*out_width)
    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels for all samples
    im_flat = im.reshape((-1, channels))
    Ia = im_flat[idx_a]
    Ib = im_flat[idx_b]
    Ic = im_flat[idx_c]
    Id = im_flat[idx_d]

    # calculate interpolated values
    wa = ((x1_f-x) * (y1_f-y)).dimshuffle(0, 'x')
    wb = ((x1_f-x) * (y-y0_f)).dimshuffle(0, 'x')
    wc = ((x-x0_f) * (y1_f-y)).dimshuffle(0, 'x')
    wd = ((x-x0_f) * (y-y0_f)).dimshuffle(0, 'x')
    output = T.sum([wa*Ia, wb*Ib, wc*Ic, wd*Id], axis=0)
    return output


def _linspace(start, stop, num):
    # Theano linspace. Behaves similar to np.linspace
    start = T.cast(start, theano.config.floatX)
    stop = T.cast(stop, theano.config.floatX)
    num = T.cast(num, theano.config.floatX)
    step = (stop-start)/(num-1)
    return T.arange(num, dtype=theano.config.floatX)*step+start


def _meshgrid(height, width):
    # This function is the grid generator from eq. (1) in reference [1].
    # It is equivalent to the following numpy code:
    #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
    #                         np.linspace(-1, 1, height))
    #  ones = np.ones(np.prod(x_t.shape))
    #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    # It is implemented in Theano instead to support symbolic grid sizes.
    # Note: If the image size is known at layer construction time, we could
    # compute the meshgrid offline in numpy instead of doing it dynamically
    # in Theano. However, it hardly affected performance when we tried.
    x_t = T.dot(T.ones((height, 1)),
                _linspace(-1.0, 1.0, width).dimshuffle('x', 0))
    y_t = T.dot(_linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                T.ones((1, width)))

    x_t_flat = x_t.reshape((1, -1))
    y_t_flat = y_t.reshape((1, -1))
#    ones = T.ones_like(x_t_flat)
#    grid = T.concatenate([x_t_flat, y_t_flat, ones], axis=0)
    grid = T.concatenate([x_t_flat, y_t_flat], axis=0)
    return grid


class SubpixelLayer(Layer):
    def __init__(self, incoming,r,c, **kwargs):
        super(SubpixelLayer, self).__init__(incoming, **kwargs)
        self.r=r # Upscale factor
        self.c=c # number of output channels
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0],self.c,self.r*input_shape[2],self.r*input_shape[3])

    def get_output_for(self, input, deterministic=False, **kwargs):
        out = T.zeros((input.shape[0],self.output_shape[1],self.output_shape[2],self.output_shape[3]))
        for x in xrange(self.r): # loop across all feature maps belonging to this channel
            for y in xrange(self.r):
                out=T.set_subtensor(out[:,:,x::self.r,y::self.r],input[:,self.r*x+y::self.r*self.r,:,:])
        return out
        
class ScaleLayer(Layer):
    def __init__(self, incoming,r, **kwargs):
        super(ScaleLayer, self).__init__(incoming, **kwargs)
        self.r=r # Upscale factor
    def get_output_for(self, input, **kwargs):
        out = input*self.r
        return out
   
class ZeroLayer(Layer):
    def __init__(self, incoming,r, **kwargs):
        super(ZeroLayer, self).__init__(incoming, **kwargs)
        self.r=r # Upscale factor
    def get_output_for(self, input, **kwargs):

        out = T.set_subtensor(input[T.abs_(input)<self.r],0.0)
        return out
        
class ConvAggr(Layer):
    def __init__(self, incoming, num_channels, filter_size=3, stride=(1, 1),
                 pad='same', W=lasagne.init.HeNormal(), b=None, **kwargs):
        ensure_set_name('conv_aggr', kwargs)
        super(ConvAggr, self).__init__(incoming, **kwargs)
        self.conv = Conv(incoming, num_channels, filter_size, stride, pad=pad,
                         W=W, b=b, nonlinearity=None, **kwargs)

        # copy params
        self.params = self.conv.params.copy()

    def get_output_for(self, input, **kwargs):
        return self.conv.get_output_for(input)

    def get_output_shape_for(self, input_shape):
        return self.conv.get_output_shape_for(input_shape)
        
class TileLayer(Layer):
    def __init__(self, incoming, **kwargs):
        super(TileLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        out = T.tile(input[...,input.shape[-1]/2:input.shape[-1]/2+1],(1,1,1,1,input.shape[-1]))
        return out

class MC_prep(Layer):
    def __init__(self, incoming, **kwargs):
        super(MC_prep, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        out = T.zeros((input.shape[-1], input.shape[1], input.shape[2], input.shape[3], input.shape[4]))
        out = T.set_subtensor(out[0:1],input)
        out = T.set_subtensor(out[1:2],T.concatenate([input[...,1:2], input[...,0:1], input[..., 2:3], input[..., 3:4], input[..., 4:]], axis = 4))
        out = T.set_subtensor(out[2:3],T.concatenate([input[...,2:3], input[...,1:2], input[..., 3:4], input[..., 0:1], input[..., 4:]], axis = 4))
        out = T.set_subtensor(out[3:4],T.concatenate([input[...,3:4], input[...,2:3], input[..., 4:], input[..., 1:2], input[..., 0:1]], axis = 4))
        out = T.set_subtensor(out[4:5],T.concatenate([input[...,4:], input[...,3:4], input[..., 2:3], input[..., 1:2], input[..., 0:1]], axis = 4))
        return out

    def get_output_shape_for(self, input_shape):
        return (input_shape[-1], input_shape[1], input_shape[2], input_shape[3], input_shape[4])
