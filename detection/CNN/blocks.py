import keras
import keras.backend as K


def conv_batch_relu(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)

    def f(x):
        conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                   kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
        bn = keras.layers.BatchNormalization()(conv)
        relu = keras.layers.ReLU()(bn)
        return relu

    return f


def conv_batch_relu_1d(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)

    def f(x):
        conv = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                   kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
        bn = keras.layers.BatchNormalization()(conv)
        relu = keras.layers.ReLU()(bn)
        return relu

    return f


def bn_relu_conv(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)

    def f(x):
        bn = keras.layers.BatchNormalization()(x)
        relu = keras.layers.ReLU()(bn)
        conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                   kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(relu)
        return conv

    return f


def bn_relu_conv_1d(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)

    def f(x):
        bn = keras.layers.BatchNormalization()(x)
        relu = keras.layers.ReLU()(bn)
        conv = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                   kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(relu)
        return conv

    return f


def res_block_2d(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)
    def f(x):
        residual = bn_relu_conv(filters=filters, kernel_size=kernel_size, strides=strides,
                                kernel_initializer=kernel_initializer, padding=padding,
                                kernel_regularizer=kernel_regularizer)(x)
        residual = bn_relu_conv(filters=filters, kernel_size=kernel_size, strides=(1, 1),
                                kernel_initializer=kernel_initializer, padding=padding,
                                kernel_regularizer=kernel_regularizer)(residual)

        input_shape = K.int_shape(x)
        residual_shape = K.int_shape(residual)
        stride_width = int(round(input_shape[1] / residual_shape[1]))
        stride_height = int(round(input_shape[2] / residual_shape[2]))
        equal_channels = input_shape[3] == residual_shape[3]

        shortcut = x
        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = keras.layers.Conv2D(filters=residual_shape[3], kernel_size=(stride_width, stride_height),
                                           strides=(stride_width, stride_height), padding="valid",
                                           kernel_initializer="he_normal")(x)

        return keras.layers.Add()([shortcut, residual])

    return f


def res_block_1d(**conv_params):

    def f(x):
        F = bn_relu_conv_1d(**conv_params)(x)
        F = bn_relu_conv_1d(**conv_params)(F)

        input_shape = K.int_shape(x)
        residual_shape = K.int_shape(F)
        stride_width = int(round(input_shape[1] / residual_shape[1]))
        equal_channels = input_shape[2] == residual_shape[2]

        shortcut = x
        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or not equal_channels:
            shortcut = keras.layers.Conv1D(filters=residual_shape[2], kernel_size=1, strides=stride_width,
                                           padding="valid", kernel_initializer="he_normal")(x)

        return keras.layers.Add()([shortcut, F])

    return f
