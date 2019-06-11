import numpy as np
import keras
from keras import backend as K
from skimage import filters
from skimage import transform
from skimage import util


def get_gradients(df, dx, feed_dict):
    grad_tensor = K.gradients(df, dx)
    sess = K.get_session()
    grad = sess.run(grad_tensor, feed_dict)[0]
    return grad


def get_layer_output(model, layer, x, batch_size=64):
    if isinstance(layer, int):
        layer = model.layers[layer]
    elif isinstance(layer, keras.layers.Layer):
        pass
    else:
        raise Exception("layer argument must be an integer layer number or a keras layer")

    layer_output_func = K.function(model.inputs, [layer.output])
    output_shape = [s.value for s in layer.output.shape]
    output_shape[0] = x[0].shape[0]
    layer_output = np.empty(output_shape)
    for i in range(int(np.ceil(output_shape[0] / batch_size))):
        cbs = min(batch_size, output_shape[0] - i * batch_size)
        x_cb = [u[i * batch_size:i * batch_size + cbs] for u in x]
        output_cb = layer_output_func(x_cb)[0]
        layer_output[i * batch_size:i * batch_size + cbs] = output_cb
    return layer_output


def feature_visualization(feature, inp, steps):
    """
    Images were optimized for 2560 steps in a color-decorrelated fourier-transformed space, using Adam at a learning
    rate of 0.05. We used each of following transformations in the given order at each step of the optimization:

    • Padding the input by 16 pixels to avoid edge artifacts
    • Jittering by up to 16 pixels
    • Scaling by a factor randomly selected from this list: 1, 0.975, 1.025, 0.95, 1.05
    • Rotating by an angle randomly selected from this list; in degrees: -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5
    • Jittering a second time by up to 8 pixels
    • Cropping the padding
    """
    shape = [d._value for d in inp.shape]
    shape[0] = 1
    x = np.random.randn(*shape) * .1

    sess = K.get_session()
    grad = K.gradients(feature, inp)

    for i in range(steps):
        g = sess.run(grad, {inp: x})[0]
        g = filters.gaussian(g, sigma=1, multichannel=True, preserve_range=True)
        o_size = g.shape
        g = transform.rescale(g, np.random.choice([1, 0.975, 1.025, 0.95, 1.05]), multichannel=True)
        g = util.crop(g, [g.shape[0] - o_size[0], g.shape[1] - o_size[1], g.shape[2] - o_size[2]])
        x += g
    return x
