import tensorflow as tf

layers = tf.keras.layers


def _get_keras_update_ops(model):
    """Gets a list of update_ops of keras model
    """
    update_ops = []
    for op in model.updates:
        if isinstance(op, tuple):
            update_ops.append(tf.assign(op[0], op[1]))
        else:
            update_ops.append(op)
    return update_ops


def _get_keras_regularizers(model):
    """Gets a list of update_ops of keras model
    """
    regularizers = None
    if len(model.losses) > 0:
        regularizers = tf.add_n(model.losses)
    return regularizers


def lenet(outputs):
    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, (3, 3), kernel_initializer='glorot_uniform')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), kernel_initializer='glorot_uniform')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = layers.Dropout(0.5)(x)

    ys = []
    for output in outputs:
        ys.append(layers.Dense(output['num'], activation='softmax',
            kernel_initializer='glorot_uniform')(x))
    model = tf.keras.models.Model(inputs=[inputs], outputs=ys)

    return (model.inputs, model.outputs,
            _get_keras_update_ops(model), _get_keras_regularizers(model))


def alexnet(outputs):
    img_in = layers.Input(shape=(32, 32, 3))
    x = layers.Convolution2D(
        32, (3, 3), kernel_regularizer=None, padding='same')(img_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Convolution2D(
        32, (3, 3), kernel_regularizer=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Convolution2D(
        64, (3, 3), kernel_regularizer=None, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Convolution2D(
        64, (3, 3), kernel_regularizer=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Convolution2D(
        128, (3, 3), kernel_regularizer=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Convolution2D(
        128, (3, 3), kernel_regularizer=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)
    ys = []
    for output in outputs:
        ys.append(layers.Dense(
            output['num'], activation='softmax')(x))
    model = tf.keras.models.Model(inputs=[img_in], outputs=ys)

    return (model.inputs, model.outputs,
            _get_keras_update_ops(model), _get_keras_regularizers(model))


def resnet_20(outputs):
    def residual_block(x, o_filters, block_count, increase=False):
        stride = (1, 1) if not increase else (2, 2)
        with tf.variable_scope('block_{}'.format(block_count)):
            o1 = layers.Activation('relu')(layers.BatchNormalization()(x))
            conv_1 = layers.Conv2D(
                o_filters, kernel_size=(3, 3), strides=stride, padding='same',
                kernel_initializer="he_normal")(o1)
            o2 = layers.Activation('relu')(layers.BatchNormalization()(conv_1))
            conv_2 = layers.Conv2D(
                o_filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                kernel_initializer="he_normal")(o2)
            if increase:
                projection = layers.Conv2D(
                    o_filters, kernel_size=(1, 1), strides=(2, 2), padding='same',
                    kernel_initializer="he_normal")(o1)
                block = layers.add([conv_2, projection])
            else:
                block = layers.add([conv_2, x])
        return block

    img_input = layers.Input(shape=(32, 32, 3))
    stack_n = 2
    x = layers.Conv2D(
        filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
        kernel_initializer="he_normal")(img_input)

    block_count = 1
    for _ in range(stack_n):
        block_count += 1
        x = residual_block(x, 16, block_count, False)

    block_count += 1
    x = residual_block(x, 32, block_count, True)
    for _ in range(1, stack_n):
        block_count += 1
        x = residual_block(x, 32, block_count, False)

    block_count += 1
    x = residual_block(x, 64, block_count, True)
    for _ in range(1, stack_n):
        block_count += 1
        x = residual_block(x, 64, block_count, False)

    ys = []
    for output in outputs:
        y = layers.BatchNormalization()(x)
        y = layers.Activation('relu')(y)
        y = layers.GlobalAveragePooling2D()(y)
        ys.append(layers.Dense(
            output['num'], activation='softmax',
            kernel_initializer='he_normal')(y))
    model = tf.keras.models.Model(inputs=[img_input], outputs=ys)

    return (model.inputs, model.outputs,
            _get_keras_update_ops(model), _get_keras_regularizers(model))


def mobilenetv2_mtc(num_outputs=[1000],
                    freeze_till='out_relu',
                    alpha=1.0,
                    size=224):
    """Loads a MobileNetV2 network for multiple classification tasks (MTC)
    """
    bottom = tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape=(size, size, 3), alpha=alpha, include_top=False)

    x = bottom.outputs[0]
    x = layers.GlobalAveragePooling2D()(x)

    outputs = []
    for i in range(len(num_outputs)):
        y = layers.Dense(num_outputs[i], activation='softmax')(x)
        outputs.append(y)

    model = tf.keras.models.Model(inputs=bottom.inputs, outputs=outputs)

    return (bottom.inputs, outputs,
            _get_keras_update_ops(model), _get_keras_regularizers(model))
