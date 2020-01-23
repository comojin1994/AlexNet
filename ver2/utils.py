import tensorflow as tf
import args

# Funtion
def get_label(path):
    fname = tf.strings.split(path, '_')[-1]
    lbl_name = tf.strings.regex_replace(fname, '.png', '')
    return lbl_name

def onehot_encoding(image_label):
    onehot_encoding = tf.cast(args.classes == image_label, tf.uint8)
    return onehot_encoding

def read_image_label(path):
    gfile = tf.io.read_file(path)
    image = tf.io.decode_image(gfile)

    image = tf.cast(image, tf.float32) / 255.

    label = get_label(path)
    label = onehot_encoding(label)

    return image, label

def image_preprocessing(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    return image, label

def scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))