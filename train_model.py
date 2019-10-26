# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:01:23 2019

@author: rwalr
"""
import sys
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint 
import tensorflow as tf 

from models import model_1d, model_2d, model_1d_rnn, model_1d_rnn_2
from seq2mat import DataGenerator, seq_class, label_dict

class_dict = {val: key for key, val in label_dict.items()}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


train_val, test, y_train_val, y_test = train_test_split(seq_class.index, seq_class['label'], test_size=0.1, random_state=42)
train, validation, y_train, y_val = train_test_split(train_val, y_train_val, test_size=0.2, random_state=42)
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
class_weights = dict(enumerate(class_weights))

def train_model(model, params, checkpoint, epochs, train_data, validation_data, labels, 
                loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'], **kwargs):
    """Function to train models using DataGenerator defined in seq2mat. 
    args:
        model: model to train
        params: paramter dictionary for DataGenerator
        checkpoint: file name for checkpointed
        epochs: number of epochs to train
        train_data: training data
        validation data: validation data
        labels: labels for all data
    """

    print(model.summary())

    training_generator = DataGenerator(train_data, labels, **params)
    validation_generator = DataGenerator(validation_data, labels, **params)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    checkpointer = ModelCheckpoint(filepath=checkpoint, 
                               verbose=1, save_best_only=True)
    hist = model.fit_generator(
        generator=training_generator, 
        validation_data=validation_generator,
        epochs=epochs,
        callbacks = [checkpointer],
        **kwargs
    )

    return hist

if __name__ == '__main__':
    # Train 1d_rnn
    train_model(
        model=model_1d_rnn,
        params={
            'batch_size': 40, 
            'dim': (1502, 31),
            'n_classes': 25, 
            'shuffle': True,
            'struct': '1d',
            'random_insert': False,
            'rnn': True
        },
        checkpoint='model_1d_rnn.model.best.hdf5',
        epochs=5,
        train_data=train,
        validation_data=validation,
        labels=seq_class['label'],
        use_multiprocessing=True,
        workers=4,
        class_weight=class_weights,
        verbose=1
    )

    # train 1d_rnn_2
    train_model(
        model=model_1d_rnn_2,
        params={
            'batch_size': 32, 
            'dim': (1502, 36),
            'n_classes': 25, 
            'shuffle': True,
            'struct': '2d',
            'random_insert': False,
            'rnn': True
        },
        checkpoint='model_1d_rnn_2.model.best.hdf5',
        epochs=5,
        train_data=train,
        validation_data=validation,
        labels=seq_class['label'],
        use_multiprocessing=True,
        workers=4,
        class_weight=class_weights,
        verbose=1
    )

    # train 2d
    train_model(
        model=model_2d,
        params={
            'batch_size': 24, 
            'dim': (3072, 8, 6),
            'n_classes': 25, 
            'shuffle': True,
            'struct': '2d'
        },
        checkpoint='model_2d.model.best.hdf5',
        epochs=5,
        train_data=train,
        validation_data=validation,
        labels=seq_class['label'],
        use_multiprocessing=True,
        workers=4,
        class_weight=class_weights,
        verbose=1
    )

    # train 1d
    train_model(
        model=model_1d,
        params={
            'batch_size': 128, 
            'dim': (1502, 31),
            'n_classes': 25, 
            'shuffle': True,
            'struct': '1d',
            'random_insert': False
        },
        checkpoint='model_1d.model.best.hdf5',
        epochs=5,
        train_data=train,
        validation_data=validation,
        labels=seq_class['label'],
        use_multiprocessing=True,
        workers=4,
        class_weight=class_weights,
        verbose=1
    )