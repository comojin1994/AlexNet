import tensorflow as tf
import args

from layers import *

def model():
    inputs = input(shape=args.input_shape)

    net = conv2d(48, 3, 1, 'SAME', 'relu')(inputs)
    net = maxpool2d(2, 2, 'SAME')(net)

    net = conv2d(128, 3, 1, 'SAME', 'relu')(net)
    net = maxpool2d(2, 2, 'SAME')(net)

    net = conv2d(192, 3, 1, 'SAME', 'relu')(net)
    net = conv2d(192, 3, 1, 'SAME', 'relu')(net)
    net = conv2d(192, 3, 1, 'SAME', 'relu')(net)
    net = maxpool2d(2, 2, 'SAME')(net)

    net = flatten()(net)
    net = dense(2048, 'relu')(net)
    net = dense(2048, 'relu')(net)
    net = dense(10, 'softmax')(net)

    model = tf.keras.Model(inputs=inputs, outputs=net, name='AlexNet')

    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# print(model.summary())
