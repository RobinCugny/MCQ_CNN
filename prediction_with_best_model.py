#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:25:00 2019

@author: robin
"""

from QCM_utils import load_data, preprocessing
from keras.models import load_model
import numpy as np

X,Y,classes=load_data()
nb_classes=len(classes)
X,Y=preprocessing(X,Y,nb_classes)
model = load_model('results/best_satisfaction_classifier.h5')
prediction=model.predict(X[:5])
prediction=np.argmax(prediction,axis=1)

