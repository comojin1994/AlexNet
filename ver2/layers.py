import tensorflow as tf
from tensorflow.keras import layers

def input(shape, dtype=tf.float32):
    Input = layers.Input(shape, dtype=dtype)
    return Input

def conv2d(filters, kernel_size, strides, padding, activation):
    Conv2d = layers.Conv2D(filters, kernel_size, strides,
                           padding=padding, activation=activation)
    return Conv2d

def maxpool2d(pool_size, stride, padding):
    Maxpool2d = layers.MaxPool2D(pool_size, stride, padding=padding)
    return Maxpool2d

def flatten():
    return layers.Flatten()

def dense(node, activation):
    Dense = layers.Dense(node, activation)
    return Dense

def dropout(rate):
    Dropout = layers.Dropout(rate)
    return Dropout