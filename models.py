# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 08:40:59 2019

@author: rwalr
"""

import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Permute, Dense, Flatten, Dropout, SpatialDropout2D
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Permute, Dense, SeparableConv1D, SpatialDropout1D

from seq2mat import DataGenerator, seq_class

n = 64
model_1d = Sequential()
model_1d.add(SeparableConv1D(n, 3, activation='relu', input_shape=(2000,31)))
model_1d.add(SeparableConv1D(n, 3, activation='relu'))
model_1d.add(MaxPooling1D(2))
model_1d.add(SpatialDropout1D(rate=0.2))

n = 128
model_1d.add(SeparableConv1D(n, 3, activation='relu'))
model_1d.add(SeparableConv1D(n, 3, activation='relu'))
model_1d.add(MaxPooling1D(2))
model_1d.add(SpatialDropout1D(rate=0.2))

n = 256
model_1d.add(SeparableConv1D(n, 3, activation='relu'))
model_1d.add(SeparableConv1D(n, 3, activation='relu'))
model_1d.add(MaxPooling1D(4))
model_1d.add(SpatialDropout1D(rate=0.2))

n = 512
model_1d.add(SeparableConv1D(n, 3, activation='relu'))
model_1d.add(SeparableConv1D(n, 3, activation='relu'))
model_1d.add(MaxPooling1D(4))
model_1d.add(SpatialDropout1D(rate=0.2))


n = 1024
model_1d.add(SeparableConv1D(n, 3, activation='relu'))
model_1d.add(MaxPooling1D(8))
model_1d.add(SpatialDropout1D(rate=0.2))

model_1d.add(Flatten())

model_1d.add(Dense(512, activation='relu'))
model_1d.add(Dropout(rate=0.2))
model_1d.add(Dense(512, activation='relu'))
model_1d.add(Dropout(rate=0.2))
model_1d.add(Dense(512, activation='relu'))
model_1d.add(Dropout(rate=0.2))

model_1d.add(Dense(128, activation='relu'))
model_1d.add(Dropout(rate=0.2))
model_1d.add(Dense(25, activation='softmax'))


