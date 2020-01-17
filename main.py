import tensorflow as tf
from tensorflow.keras import datasets
import Hyperparam as hp
import model as m

cifar10 = datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255.
x_test = x_test / 255.

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(1000)
train_ds = train_ds.batch(hp.batch_size)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(hp.batch_size)

for epoch in range(hp.epoch):
    print('Start Training')
    for images, labels in train_ds:
        m.train_step(images, labels)

    for test_images, test_labels in test_ds:
        m.test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1, m.train_loss.result(),
                          m.train_accuracy.result() * 100,
                          m.test_loss.result(),
                          m.test_accuracy.result() * 100))
print('End Training')

m.model.evaluate(x_test, y_test, batch_size=hp.batch_size)