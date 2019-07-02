STEPS_PER_EPOCH = 16

print(len(sys.argv))
print("hello world")

import keras.utils as ku
import keras.callbacks as KC
import matplotlib.pyplot as plt
import numpy as np

from data import BeeLoader
from model import build_model

bee_data = BeeLoader()
bee_model = build_model()

def my_generator(batch_size):
  
  # TODO, probably don't do this
  foo = np.zeros((128,))
  
  # TODO, what the ids?
  for images, ids in bee_data.generator(batch_size):
    
    yield images, foo

checkpt = KC.ModelCheckpoint('./existing_dir/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=False, 
                             ave_weights_only=True,
                             mode='auto',
                             period=1)

bee_model.fit_generator(my_generator(6),
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=1,
                        verbose=1,
                        callbacks=[checkpt],
                        validation_data=None,
                        validation_steps=None,
                        validation_freq=1,
                        class_weight=None,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False,
                        shuffle=False,
                        initial_epoch=0)
