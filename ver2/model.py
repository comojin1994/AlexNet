import tensorflow as tf
from layers import input, conv2d, maxpool2d, flatten, dense, dropout

input_shape = (32, 32, 3)
inputs = input(shape=input_shape)

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

print(model.summary())