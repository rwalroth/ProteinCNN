# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 19:41:03 2019

@author: rwalr
"""
from random import randrange

import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

import numpy as np
from tensorflow import keras

# Load in the processed data and the keys for amino acids
seq_class = pd.read_csv('data\\Seq_class.csv', index_col='structureId')
aa_keys = pd.read_csv('data\\AA_keys.csv', index_col='One Letter')

# Keras expects numeric labels, this generates those
label_dict = {val:i for i, val in enumerate(set(seq_class['classification']))}
seq_class['label'] = [label_dict[key] for key in seq_class['classification']]

aa_keys_struct = {
        'R': [[2, 2, 2, 1, 2, 2, 0, 0],
              [1, 1, 1, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],
			  
        'Q': [[2, 2, 0, 2, 0, 0, 0, 0],
              [1, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],
			  
        'F': [[2, 1, 2, 2, 0, 0, 0, 0],
              [1, 2, 2, 2, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 1, 1, 0, 0, 0, 0]],
			  
        'Y': [[2, 1, 2, 2, 1, 0, 0, 0],
              [1, 2, 2, 2, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 1, 1, 0, 0, 0, 0]],
			  
        'W': [[2, 1, 1, 2, 2, 0, 0, 0],
              [1, 2, 2, 2, 2, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 1, 1, 1, 0, 0, 0]],
			  
        'K': [[2, 2, 2, 2, 2, 0, 0, 0],
              [1, 1, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],
			  
        'G': [[0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],
			  
        'A': [[3, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],
			  
        'H': [[2, 0, 3, 0, 0, 0, 0, 0],
              [1, 1, 2, 0, 0, 0, 0, 0],
              [0, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 1, 0, 0, 0, 0, 0]],
			  
        'S': [[2, 1, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],
			  
        'P': [[6, 0, 0, 0, 0, 0, 0, 0],
              [3, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0]],
			  
        'E': [[2, 2, 0, 1, 0, 0, 0, 0],
              [1, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],
			  
        'D': [[2, 0, 1, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],
			  
        'T': [[2, 3, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],
			  
        'C': [[2, 1, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],
			  
        'M': [[2, 2, 0, 3, 0, 0, 0, 0],
              [1, 1, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],
			  
        'L': [[2, 4, 3, 0, 0, 0, 0, 0],
              [1, 2, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],
			  
        'N': [[2, 0, 2, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],
			  
        'I': [[4, 2, 3, 0, 0, 0, 0, 0],
              [2, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],
			  
        'V': [[4, 3, 0, 0, 0, 0, 0, 0],
              [2, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],
        }

def aa_to_map(aa):
      """Takes in one letter code for amino acid and returns a 
      two dimensional represenation of the amino acid
      """
      map = np.array(aa_keys_struct[aa]).T
      out = np.zeros((1, 8, 6))
      out[0] = map
      return out

def get_mat_1d(sequence, dim):
      """Takes in sequnce and dimension, returns a full matrix representation
      of the sequence using the aa_keys DataFrame for each amino
      acid in the sequence
      """
      mat = np.array([aa_keys.loc[aa] for aa in sequence])

      # Allows for Conv2D or Conv1D layers
      if len(dim) == 2:
            mat = mat.reshape((mat.shape[0], mat.shape[1]))
      elif len(dim) == 3:
            mat = mat.reshape((mat.shape[0], mat.shape[1], 1))
      else:
            print("Incompatible Dimensions")
            mat = 0
      return mat

def get_mat_2d(sequence, rnn=False):
      """Uses aa_to_map to turn a sequence into a 3D array representation of the
      protein.
      """
      if rnn:
            mat = np.zeros((len(sequence), 36))
            for i, aa in enumerate(sequence):
                  mat[i] = aa_to_map(aa)[:,:6,:].flatten()
      else:
            mat = np.zeros((2*len(sequence), 8, 6))
            for i, aa in enumerate(sequence):
                  mat[2*i] = aa_to_map(aa)
      return mat

class DataGenerator(keras.utils.Sequence):
      'Generates data for Keras'
      def __init__(self, list_IDs, labels, batch_size=32, dim=(2000, 31), struct='1d',
                  n_classes=25, shuffle=True, random_insert=True, rnn=False):
            'Initialization.'
            super().__init__()
            self.dim = dim
            self.batch_size = batch_size
            self.labels = labels
            self.list_IDs = list_IDs
            self.struct = struct
            self.n_classes = n_classes
            self.shuffle = shuffle
            self.random_insert = random_insert
            self.rnn = rnn
            self.on_epoch_end()

      def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.floor(len(self.list_IDs) / self.batch_size))

      def __getitem__(self, index):
            'Generate one batch of data'
            # Generate indexes of the batch

            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

            # Find list of IDs
            list_IDs_temp = [self.list_IDs[k] for k in indexes]

            # Generate data
            X, y = self.__data_generation(list_IDs_temp)

            return X, y

      def on_epoch_end(self):
            'Updates indexes after each epoch'
            if self.rnn:
                  self.indexes = np.argsort(seq_class['sequence'][self.list_IDs].str.len())[::-1]
            else:
                  self.indexes = np.arange(len(self.list_IDs))
                  if self.shuffle == True:
                        np.random.shuffle(self.indexes)
    
      def __seq_to_mat(self, sequence, dim):
            # Make a matrix where each row is the information for the amino acid in the sequence
            if self.struct == '1d':
                  mat = get_mat_1d(sequence, dim)
            elif self.struct == '2d':
                  mat = get_mat_2d(sequence, self.rnn)
		
		# Creates a matrix of zeros, randomly inserts the relevant data
            out = np.zeros(dim)
            if self.random_insert:
                  idx = randrange(0,out.shape[0] - mat.shape[0])
            else:
                  idx = out.shape[0] - mat.shape[0]
            out[idx:mat.shape[0] + idx] = mat
            return out
        

      def __data_generation(self, list_IDs_temp):
            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
            # Initialization
            if self.rnn:
                  maxseq = seq_class['sequence'][list_IDs_temp].str.len().max()
                  dim = maxseq, self.dim[1]
            else:
                  dim = self.dim
            X = np.empty((self.batch_size, *dim))
            y = np.empty((self.batch_size), dtype=int)

            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                  # Store sample
                  seq = seq_class['sequence'].loc[ID]
                  X[i,] = self.__seq_to_mat(seq, dim)

                  # Store class
                  y[i] = self.labels[ID]
            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)	

