import tensorflow as tf
import os
import args
import callback as call

from glob import glob
from datetime import datetime

from utils import *
from layers import *
from model import *


# Model
model = model()

# training
print('Start training')
starttime = datetime.now()

history = model.fit_generator(
    generator=args.train_dataset,
    steps_per_epoch=args.train_step,
    epochs=args.num_epoch,
    validation_data=args.test_dataset,
    validation_steps=args.test_step,
    callbacks=[call.tensorboard, call.learning_rate_scheduler, call.checkpoint]

)

endtime = datetime.now()

print(history.params)
print('End training')
print((endtime-starttime))