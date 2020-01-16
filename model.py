import tensorflow as tf

from layers import Layers

input_shape = (32, 32, 3)

def model(intput_shape):
    inputs = Layers.Input(shape = input_shape)

    net = Layers.Conv2D(48, 3, 1, 'SAME', 'relu')(inputs)
    net = Layers.MaxPool2D(2, 2, 'SAME')(net)

    net = Layers.Conv2D(128, 3, 1, 'SAME', 'relu')(net)
    net = Layers.MaxPool2D(2, 2, 'SAME')(net)

    net = Layers.Conv2D(192, 3, 1, 'SAME', 'relu')(net)
    net = Layers.Conv2D(192, 3, 1, 'SAME', 'relu')(net)
    net = Layers.Conv2D(128, 3, 1, 'SAME', 'relu')(net)
    net = Layers.MaxPool2D(2, 2, 'SAME')(net)

    net = Layers.Flatten()(net)
    net = Layers.Dense(2048, 'relu')(net)
    net = Layers.Dense(2048, 'relu')(net)
    net = Layers.Dense(10, 'softmax')(net)

    model = tf.keras.Model(inputs=inputs, outputs=net, name='AlexNet')

    return model
