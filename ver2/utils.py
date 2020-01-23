import tensorflow as tf

def onehot_encoding(image_label, classes):
    onehot_encoding = tf.cast(classes == image_label, tf.uint8)
    return onehot_encoding

def read_image_label(image, label, classes):
    image = tf.cast(image, tf.float32) / 255.

    label = onehot_encoding(label, classes)

    return image, label

def image_preprocessing(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    return image, label