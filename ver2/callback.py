import tensorflow as tf
import args
from utils import scheduler

# Callback
# tensorboard
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=args.logdir,
    write_graph=True,
    write_images=True,
    histogram_freq=1
)

# learning rate scheduale
learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

# checkpoint
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    args.save_path,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True
)