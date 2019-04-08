#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 12:08:47 2019

@author: robin
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
from QCM_utils import load_data, preprocessing,split

batch_size = 4
epochs=16
proportion=0.9

X,Y,classes=load_data()
nb_classes=len(classes)
X,Y=preprocessing(X,Y,nb_classes)
X_train,X_test,Y_train,Y_test=split(X,Y,proportion)

input_shape=X_train.shape[1:]

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='results/satisfaction_classifier.h5', monitor='val_loss', save_best_only=True)]
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test),
          callbacks=callbacks)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

prediction=model.predict(X_test)

prediction=np.argmax(prediction,axis=1)

plt.figure()
for i,p in enumerate(prediction[:12]):
    plt.subplot(3,4,i+1)
    plt.axis('off')
    plt.title("Prediction : "+classes[p])
    plt.imshow(np.reshape(X_test[i],(12,100)), cmap="gray")
    plt.show() 
