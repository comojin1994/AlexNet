from tensorflow.keras import layers
import tensorflow as tf

class Layers:
    @staticmethod
    def Input(shape):
        input = layers.Input(shape, dtype=tf.float32)
        return input

    @staticmethod
    def Conv2D(filters, kernel_size, strides, padding, activation):
        conv2D = layers.Conv2D(filters, kernel_size, strides,
                               padding=padding, activation=activation)
        return conv2D

    @staticmethod
    def MaxPool2D(pool_size, stride, padding):
        maxpool2D = layers.MaxPool2D(pool_size, stride, padding=padding)
        return maxpool2D

    @staticmethod
    def Flatten():
        return layers.Flatten()

    @staticmethod
    def Dense(node, activation):
        dense = layers.Dense(node, activation)
        return dense

    @staticmethod
    def Dropout(rate):
        dropout = layers.Dropout(rate)
        return dropout