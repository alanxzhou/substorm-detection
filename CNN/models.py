import keras
from CNN import blocks
import utils


def train_strided_station_cnn(X_train, y_train, X_val, y_val, params):
    """ The concept here is that substorms will be detectable from individual stations, so this network
    processes stations individually (kernel size 1 in station dimension) and uses global max pooling to check
    if any of the stations are seeing a substorm.
    params:
        stages
        blocks_per_stage
        downsampling_per_stage
        batch_size
        epochs
        flx2
        kernel_width
        fl_filters
        fl_stride
        fl_kernel_width
    """
    model_input = keras.layers.Input(shape=X_train.shape[1:])
    net = blocks.conv_batch_relu(filters=params['fl_filters'], kernel_size=[1, params['fl_kernel_width']],
                                 strides=[1, params['fl_stride']])(model_input)

    filters = params['fl_filters']
    if params['flx2']:
        filters *= 2

    for stage in range(params['stages']):
        for _ in range(params['blocks_per_stage'] - 1):
            net = blocks.conv_batch_relu(filters=filters, kernel_size=[1, params['kernel_width']], strides=[1, 1])(net)
        net = blocks.conv_batch_relu(filters=filters, kernel_size=[1, params['kernel_width']],
                                     strides=[1, params['downsampling_per_stage']])(net)
        filters *= 2

    net = keras.layers.GlobalMaxPool2D()(net)

    model_output = keras.layers.Dense(1, activation='sigmoid')(net)

    model = keras.models.Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', utils.true_positive, utils.false_positive])
    hist = model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(X_val, y_val))

    return hist, model


def train_combiner_net(X_train, y_train, X_val, y_val, params):
    model_input = keras.layers.Input(shape=X_train.shape[1:])
    net = blocks.conv_batch_relu(filters=params['fl_filters'], kernel_size=[1, params['fl_kernel_width']],
                                 strides=[1, params['fl_stride']])(model_input)

    filters = params['fl_filters']
    if params['flx2']:
        filters *= 2

    for stage in range(params['stages']):
        for _ in range(params['blocks_per_stage'] - 1):
            net = blocks.combiner(X_train.shape[1], filters=filters, kernel_size=[1, params['kernel_width']],
                                  strides=[1, 1])(net)
        net = blocks.combiner(X_train.shape[1], filters=filters, kernel_size=[1, params['kernel_width']],
                              strides=[1, params['downsampling_per_stage']])(net)
        filters *= 2

    net = keras.layers.GlobalAveragePooling2D()(net)

    model_output = keras.layers.Dense(1, activation='sigmoid')(net)

    model = keras.models.Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', utils.true_positive, utils.false_positive])
    hist = model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(X_val, y_val))

    return hist, model


def train_resnet():
    resnet_input = keras.layers.Input(X_train.shape[1:])
    blocks = [2, 2, 2, 2]

    numerical_names = [True] * len(blocks)

    x = keras.layers.Conv2D(64, (1, 13), strides=(1, 1), use_bias=False, name="conv1", padding="same")(resnet_input)
    x = keras.layers.BatchNormalization(name="bn_conv1")(x)
    x = keras.layers.Activation("relu", name="conv1_relu")(x)
    x = keras.layers.MaxPooling2D((1, 3), strides=(2, 2), padding="same", name="pool1")(x)

    features = 64

    outputs = []

    for stage_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            x = blocks.mag_block(
                features,
                stage_id,
                block_id,
                numerical_name=(block_id > 0 and numerical_names[stage_id]),
                freeze_bn=False
            )(x)

        features *= 2

        outputs.append(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(2, activation='softmax')(x)
    model = keras.models.Model(inputs=resnet_input, outputs=x)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', true_positive, false_positive])
    resnet_hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                            class_weight=class_weight)