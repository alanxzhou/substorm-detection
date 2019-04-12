import keras
import keras_resnet


def conv_batch_relu(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)

    def f(x):
        conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
        bn = keras.layers.BatchNormalization()(conv)
        relu = keras.layers.ReLU()(bn)
        return relu

    return f


def mag_block(
    filters,
    stage=0,
    block=0,
    kernel_size=[1, 7],
    numerical_name=False,
    stride=None,
    freeze_bn=False
):
    """
    A two-dimensional basic block.
    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    Usage:
        >>> import keras_resnet.blocks
        >>> keras_resnet.blocks.basic_2d(64)
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = [1, 2]

    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = keras.layers.Conv2D(filters, kernel_size, strides=stride, use_bias=False, name="res{}{}_branch2a".format(stage_char, block_char), kernel_initializer='he_normal', padding='same')(x)

        y = keras_resnet.layers.BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2a".format(stage_char, block_char))(y)

        y = keras.layers.Activation("relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = keras.layers.Conv2D(filters, kernel_size, use_bias=False, name="res{}{}_branch2b".format(stage_char, block_char), kernel_initializer='he_normal', padding='same')(y)

        y = keras_resnet.layers.BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2b".format(stage_char, block_char))(y)

        if block == 0:
            shortcut = keras.layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch1".format(stage_char, block_char), kernel_initializer='he_normal')(x)

            shortcut = keras_resnet.layers.BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch1".format(stage_char, block_char))(shortcut)
        else:
            shortcut = x

        y = keras.layers.Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])

        y = keras.layers.Activation("relu", name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f