# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 08:40:59 2019

@author: rwalr
"""

from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, 
    SeparableConv2D, 
    MaxPooling2D, 
    Permute, 
    Dense, 
    Flatten, 
    Dropout, 
    SpatialDropout2D, 
    Conv1D, 
    MaxPooling1D, 
    Permute, 
    Dense, 
    SeparableConv1D, 
    SpatialDropout1D, 
    ConvLSTM2D,
    LSTM
)

from seq2mat import DataGenerator, seq_class

##################################################################################
#                           1D                                                   #
##################################################################################

n = 128
model_1d = Sequential()
model_1d.add(SeparableConv1D(n, 5, activation='relu', padding='same', input_shape=(1502,31)))
model_1d.add(SeparableConv1D(n, 3, activation='relu', padding='same'))
model_1d.add(MaxPooling1D(2))
model_1d.add(SpatialDropout1D(rate=0.2))

n = 128
model_1d.add(SeparableConv1D(n, 3, activation='relu', padding='same'))
model_1d.add(SeparableConv1D(n, 3, activation='relu', padding='same'))
model_1d.add(MaxPooling1D(2))
model_1d.add(SpatialDropout1D(rate=0.2))

n = 256
model_1d.add(SeparableConv1D(n, 3, activation='relu', padding='same'))
model_1d.add(SeparableConv1D(n, 3, activation='relu', padding='same'))
model_1d.add(MaxPooling1D(4))
model_1d.add(SpatialDropout1D(rate=0.2))

n = 512
model_1d.add(SeparableConv1D(n, 3, activation='relu', padding='same'))
model_1d.add(SeparableConv1D(n, 3, activation='relu', padding='same'))
model_1d.add(MaxPooling1D(4))
model_1d.add(SpatialDropout1D(rate=0.2))


n = 1024
model_1d.add(SeparableConv1D(n, 3, activation='relu', padding='same'))
model_1d.add(MaxPooling1D(16))
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

##################################################################################
#                           2D                                                   #
##################################################################################

n = 64
model_2d = Sequential()
model_2d.add(Conv2D(n, (6,2), strides=(1,1), padding='same', activation='relu', input_shape=(3072,8,6)))
model_2d.add(SeparableConv2D(n, (3,2), padding='same', activation='relu'))
model_2d.add(MaxPooling2D((6,1)))
model_2d.add(SpatialDropout2D(rate=0.2))

n = 128
model_2d.add(SeparableConv2D(n, (3,2), padding='same', activation='relu'))
model_2d.add(MaxPooling2D((8,2)))
model_2d.add(SpatialDropout2D(rate=0.2))

n = 256
model_2d.add(SeparableConv2D(n, (2,2), padding='same', activation='relu'))
model_2d.add(MaxPooling2D((4,1)))
model_2d.add(SpatialDropout2D(rate=0.2))

n = 512
model_2d.add(SeparableConv2D(n, 2, padding='same', activation='relu'))
model_2d.add(MaxPooling2D((8,2)))
model_2d.add(SpatialDropout2D(rate=0.2))


n = 1024
model_2d.add(SeparableConv2D(n, 2, padding='same', activation='relu'))
model_2d.add(MaxPooling2D((2,2)))
model_2d.add(SpatialDropout2D(rate=0.2))

model_2d.add(Flatten())

model_2d.add(Dense(512, activation='relu'))
model_2d.add(Dropout(rate=0.2))
model_2d.add(Dense(512, activation='relu'))
model_2d.add(Dropout(rate=0.2))
model_2d.add(Dense(512, activation='relu'))
model_2d.add(Dropout(rate=0.2))

model_2d.add(Dense(128, activation='relu'))
model_2d.add(Dropout(rate=0.2))
model_2d.add(Dense(25, activation='softmax'))


##################################################################################
#                           RNN                                                  #
##################################################################################

model_1d_rnn = Sequential()
model_1d_rnn.add(LSTM(64, activation='relu', input_shape=(None,31)))
model_1d_rnn.add(Dropout(rate=0.2))

model_1d_rnn.add(Flatten())

model_1d_rnn.add(Dense(128, activation='relu'))
model_1d_rnn.add(Dropout(rate=0.2))
model_1d_rnn.add(Dense(256, activation='relu'))
model_1d_rnn.add(Dropout(rate=0.2))
model_1d_rnn.add(Dense(512, activation='relu'))
model_1d_rnn.add(Dropout(rate=0.2))
model_1d_rnn.add(Dense(512, activation='relu'))
model_1d_rnn.add(Dropout(rate=0.2))
model_1d_rnn.add(Dense(512, activation='relu'))
model_1d_rnn.add(Dropout(rate=0.2))

model_1d_rnn.add(Dense(128, activation='relu'))
model_1d_rnn.add(Dropout(rate=0.2))
model_1d_rnn.add(Dense(25, activation='softmax'))

##################################################################################
#                           RNN 2                                                #
##################################################################################

model_1d_rnn_2 = Sequential()
model_1d_rnn_2.add(LSTM(64, activation='relu', input_shape=(None,6*6)))
model_1d_rnn_2.add(Dropout(rate=0.2))

model_1d_rnn_2.add(Flatten())

model_1d_rnn_2.add(Dense(128, activation='relu'))
model_1d_rnn_2.add(Dropout(rate=0.2))
model_1d_rnn_2.add(Dense(256, activation='relu'))
model_1d_rnn_2.add(Dropout(rate=0.2))
model_1d_rnn_2.add(Dense(512, activation='relu'))
model_1d_rnn_2.add(Dropout(rate=0.2))
model_1d_rnn_2.add(Dense(512, activation='relu'))
model_1d_rnn_2.add(Dropout(rate=0.2))
model_1d_rnn_2.add(Dense(512, activation='relu'))
model_1d_rnn_2.add(Dropout(rate=0.2))

model_1d_rnn_2.add(Dense(128, activation='relu'))
model_1d_rnn_2.add(Dropout(rate=0.2))
model_1d_rnn_2.add(Dense(25, activation='softmax'))


if __name__ == '__main__':
    print(model_1d.summary())
    print(model_2d.summary())
    print(model_1d_rnn.summary())
    print(model_1d_rnn_2.summary())
