import tensorflow as tf
import os

from glob import glob
from utils import *
from datetime import datetime

# Hyperparameter
num_epoch = 1
batch_size = 32
input_shape = (32, 32, 3)
learning_rate = 0.001
dropout_rate = 0.5
num_classes = 10

# path
train_dir = r'dataset\cifar\train\*.png'
test_dir = r'dataset\cifar\test\*.png'
logdir = os.path.join(r'logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
save_path = r'checkpoints'

# load data
train_data = glob(train_dir)[:100]
test_data = glob(test_dir)[:100]

train_class = [get_label(path) for path in train_data]
test_class = [get_label(path) for path in test_data]

classes = tf.unique(train_class).y.numpy()

# make dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
train_dataset = train_dataset.map(read_image_label)
train_dataset = train_dataset.map(image_preprocessing)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.shuffle(len(train_data))
train_dataset = train_dataset.repeat()

test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
test_dataset = test_dataset.map(read_image_label)
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.shuffle(len(test_data))
test_dataset = test_dataset.repeat()

train_step = len(train_data) // batch_size
test_step = len(test_data) // batch_size