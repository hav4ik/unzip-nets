import tensorflow as tf


def mnist_net(num_outputs=[10]):
    a = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(a)
    x = tf.keras.layers.BatchNormalization(name='block1_conv1_bn')(x)
    x = tf.keras.layers.Activation('relu', name='block1_conv1_act')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = tf.keras.layers.BatchNormalization(name='block1_conv2_bn')(x)
    x = tf.keras.layers.Activation('relu', name='block1_conv2_act')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    outputs = []
    for i in range(len(num_outputs)):
        y = tf.keras.layers.Dense(num_outputs[i], activation='softmax')(x)
        outputs.append(y)

    model = tf.keras.models.Model(inputs=[a], outputs=outputs)
    update_ops = []
    for op in model.updates:
        if isinstance(op, tuple):
            update_ops.append(tf.assign(op[0], op[1]))
        else:
            update_ops.append(op)

    return model.inputs, outputs, update_ops


def mobilenetv2_mtc(num_outputs=[1000],
                    freeze_till='out_relu',
                    alpha=1.0,
                    size=224):
    """Loads a MobileNetV2 network for multiple classification tasks (MTC)
    """
    bottom = tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape=(size, size, 3), alpha=alpha, include_top=False)

    x = bottom.outputs[0]
    x = GlobalAveragePooling2D()(x)

    outputs = []
    for i in range(len(num_outputs)):
        y = Dense(num_outputs[i], activation='softmax')(x)
        outputs.append(y)

    model = tf.keras.models.Model(inputs=bottom.inputs, outputs=outputs)

    update_ops = []
    for op in model.updates:
        if isinstance(op, tuple):
            update_ops.append(tf.assign(op[0], op[1]))
        else:
            update_ops.append(op)

    return bottom.inputs, outputs, update_ops
