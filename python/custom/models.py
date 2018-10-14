import tensorflow as tf
layers = tf.keras.layers


def mnist_net(outputs=[{'name':'1','num':10}]):
    inputs = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, kernel_size=(3, 3))(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), activation=None)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    ys = []
    for output in outputs:
        ys.append(layers.Dense(output['num'], activation='softmax')(x))
    model = tf.keras.models.Model(inputs=[inputs], outputs=ys)

    update_ops = []
    for op in model.updates:
        if isinstance(op, tuple):
            update_ops.append(tf.assign(op[0], op[1]))
        else:
            update_ops.append(op)

    return model.inputs, model.outputs, update_ops


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
