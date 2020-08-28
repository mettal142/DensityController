import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import DensityController as DC

def TrainGraph():
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'],'b-',label='loss')
    plt.plot(history.history['val_loss'],'r--',label='val_loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'],'g-',label='accuracy')
    plt.plot(history.history['val_accuracy'],'k--',label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylim(0.7,1)
    plt.legend()
    plt.show()

def CheckResalt():
    for i in range(100):
       te1 = np.argmax(model.predict(np.array(testX[i]).reshape(1,3,300,1)))
       te2 = testY[i]
       if te1==te2:
           print(te1, te2, 'Right Answer', i)       
       else:
           print('End' ,i)
           break
        
(trainX,trainY),(testX,testY)=DC.DataIO.DataRead('CombinedData2100_200828',3,300,True)

trainX=np.array(trainX)
trainY=np.array(trainY)
testX=np.array(testX)
testY=np.array(testY)

ChechPoint_FileName='CheckPoint.ckpt'

CheckPoint_Callback=tf.keras.callbacks.ModelCheckpoint(filepath=ChechPoint_FileName,
                                   save_weights_only=True,
                                   verbose=1)
model = tf.keras.Sequential([tf.keras.layers.Conv2D(input_shape=(3,300,1),kernel_size=(1,3),filters=16,padding='same',activation='relu'),
                             tf.keras.layers.Conv2D(kernel_size=(1,3),padding='same',filters=32,activation='relu'),
                             tf.keras.layers.Conv2D(kernel_size=(1,3),padding='valid',filters=64,activation='relu'),
                             tf.keras.layers.Dropout(rate=0.8),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(units=128, activation='relu'),
                             tf.keras.layers.Dense(units=11, activation = 'softmax')
                             ])
model.compile(optimizer= tf.keras.optimizers.Adam(),
              loss= 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit((trainX),(trainY),epochs=30,validation_split=0.10)

model.save('GestureRecognitionModel200828_4.h5')

#TrainGraph()

resalt=model.evaluate((testX),(testY),verbose=1)


print('loss :', resalt[0], 'correntness:',resalt[1]*100,"%")

DC.DataIO.GetTest(model)