import tensorflow as tf
import Hyperparam as hp
from layers import Layers

input_shape = (32, 32, 3)
# input_shape = images[0].shape

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


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    loss = loss_object(labels, predictions)

    test_loss(loss)
    test_accuracy(labels, predictions)

# print(model(input_shape).summary())