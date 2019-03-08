import theano.tensor as T


def categorical_crossentropy(prediction, truth):
    # prediction: label probability prediction, 4D tensor
    # truth: ground truth label map, 1-hot representation, 4D tensor
    return - T.mean(T.sum(truth * T.log(prediction), axis=1))


def weighted_categorical_crossentropy(prediction, truth, weight):
    # prediction: label probability prediction, 4D tensor
    # truth: ground truth label map, 1-hot representation, 4D tensor
    return - T.mean(T.sum((truth * T.log(prediction)) * weight, axis=1))


def categorical_accuracy(prediction, truth):
    A = T.argmax(prediction, axis=1, keepdims=True)
    B = T.argmax(truth, axis=1, keepdims=True)
    return T.mean(T.eq(A, B))


def categorical_dice(prediction, truth, k):
    # Dice overlap metric for label value k
    A = T.eq(T.argmax(prediction, axis=1, keepdims=True), k)
    B = T.eq(T.argmax(truth, axis=1, keepdims=True), k)
    return 2 * T.sum(A * B) / T.cast(T.sum(A) + T.sum(B), 'float32')


def huber_loss(x):
    d_x = x[:,:,:,1:]-x[:,:,:,:-1]
    d_y = x[:,:,1:,:]-x[:,:,:-1,:]
    err = (d_x**2).sum()+(d_y**2).sum()
    err /= 20.0
    tv_err = T.sqrt(0.01+err)
    return tv_err
