# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:01:23 2019

@author: rwalr
"""
import sys
import os

import numpy as np

from models import model_1d
from seq2mat import DataGenerator, seq_class
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint 
import tensorflow as tf 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


train_val, test, y_train_val, y_test = train_test_split(seq_class.index, seq_class['label'], test_size=0.1)
train, validation, y_train, y_val = train_test_split(train_val, y_train_val, test_size=0.2)
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
class_weights = dict(enumerate(class_weights))

def train_1d(load):
    print(model_1d.summary())
    params = {
        'batch_size': 32, 
        'dim': (2000, 31),
        'n_classes': 25, 
        'shuffle': True,
        'struct': '1d'
    }

    training_generator = DataGenerator(train, seq_class['label'], **params)
    validation_generator = DataGenerator(validation, seq_class['label'], **params)

    model_1d.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
                  metrics=['accuracy'])
    
    if load:
        print('Loading')
        model_1d.load_weights('prot_6.model.best.hdf5')
    
    checkpointer = ModelCheckpoint(filepath='prot_6.model.best.hdf5', 
                               verbose=1, save_best_only=True)
    hist = model_1d.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=25,
                        callbacks = [checkpointer],
                        verbose=1,
                        use_multiprocessing=True,
                        workers=4,
                        class_weight=class_weights)

def train_2d():
    pass






if __name__ == '__main__':
    os.environ["PYTHONWARNINGS"] = "ignore"
    if sys.argv[1] == '1d':
        train_1d(sys.argv[2])
    elif sys.argv[1] == '2d':
        train_2d()