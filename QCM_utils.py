#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:10:00 2019

@author: robin
"""
import cv2
import os
import glob
import numpy as np
from sklearn.utils import shuffle
from keras.utils import np_utils

def load_data(path="data/"):
    img_dir = ["tres_satisfait","satisfait","peu_satisfait","pas_satisfait"]
    X = []
    Y = []
    for idx,folder in enumerate(img_dir):
        data_path = os.path.join(path+folder,'*g')
        files = glob.glob(data_path)
        cpt=0
        for f1 in files:
            cpt+=1
            img = cv2.imread(f1,0)
            imgf=np.flip(img) #compensation du déséquilibre des classes
            X.append(img)
            Y.append(idx)
            X.append(imgf)
            Y.append(len(img_dir)-idx-1)
            if cpt==100:
                break
                  
    folder = "cases_vides"
    
    data_path = os.path.join(path+folder,'*g')
    files = glob.glob(data_path)
    cpt=0
    for f1 in files:
        cpt+=1
        img = cv2.imread(f1,0)
        imgf1=np.flip(img)
        imgf2=np.flip(img,axis=0)
        imgf3=np.flip(imgf1,axis=0)
        X.append(img)
        X.append(imgf1)
        X.append(imgf2)
        X.append(imgf3)
        Y.append(4)
        Y.append(4)
        Y.append(4)
        Y.append(4)
        if cpt==50:
            break
    
    folder = "double_coche"
    
    data_path = os.path.join(path+folder,'*g')
    files = glob.glob(data_path)
    for f1 in files:
        img = cv2.imread(f1,0)
        imgf1=np.flip(img)
        imgf2=np.flip(img,axis=0)
        imgf3=np.flip(imgf1,axis=0)
        X.append(img)
        X.append(imgf1)
        X.append(imgf2)
        X.append(imgf3)
        Y.append(5)
        Y.append(5)
        Y.append(5)
        Y.append(5)
    
    classes = ["tres_satisfait","satisfait","peu_satisfait","pas_satisfait","cases_vides", "double_coche"]

    return X,Y,classes

def preprocessing(X,Y,nb_classes):
    X = np.asarray(X).astype('float32')
    Y = np.asarray(Y)
    X /= 255
    X, Y = shuffle(X, Y, random_state=0)
    Y = np_utils.to_categorical(Y, nb_classes)
    X=np.reshape(X,(X.shape[0],X.shape[1],X.shape[2],1))   
    return X,Y

def split(X,Y,proportion):
    X_train=X[:int(X.shape[0]*proportion),:,:]
    X_test=X[int(X.shape[0]*proportion):,:,:]
    Y_train=Y[:int(Y.shape[0]*proportion),:]
    Y_test=Y[int(Y.shape[0]*proportion):,:]
    return X_train,X_test,Y_train,Y_test