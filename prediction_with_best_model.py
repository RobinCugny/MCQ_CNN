#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:25:00 2019

@author: robin
"""

from QCM_utils import load_data, preprocessing
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

X,Y,classes=load_data()
nb_classes=len(classes)
X,Y=preprocessing(X,Y,nb_classes)
model = load_model('results/best_satisfaction_classifier.h5')
prediction=model.predict(X[:12])
prediction=np.argmax(prediction,axis=1)

plt.figure()
for i,p in enumerate(prediction):
    plt.subplot(3,4,i+1)
    plt.axis('off')
    plt.title("Prediction : "+classes[p])
    plt.imshow(np.reshape(X[i],(12,100)), cmap="gray")
    plt.show() 
