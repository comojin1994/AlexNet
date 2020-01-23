import tensorflow as tf
import tensorflow_datasets as tfds
from utils import onehot_encoding, read_image_label, image_preprocessing

train_ds = tfds.load(name='cifar10', split='train', shuffle_files=True)
test_ds = tfds.load(name='cifar10', split='test', shuffle_files=True)

train_image = [feature['image'] for feature in train_ds]
train_label = [feature['label'] for feature in train_ds]

test_image = [feature['image'] for feature in test_ds]
test_label = [feature['label'] for feature in test_ds]

classes = tf.unique(train_label).y.numpy()

train_ds = tf.data.Dataset.map(read_image_label)