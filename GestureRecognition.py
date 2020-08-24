import tensorflow as tf
import numpy as np
import DensityController as DC

model= tf.keras.models.load_model('GestureRecognitionModel.h5')
model.summary()

print('Test Start')
DC.DataIO.GetTest(model)
