import DensityController as DC

model= DC.tf.keras.models.load_model('GestureRecognitionModel200828_3.h5')
model.summary()

print('Test Start')
DC.DataIO.GetTest(model)
