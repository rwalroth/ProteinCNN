# ProteinCNN
Code for analyzing protein sequences using convolutional neural nets 

This repository contains code necessary to take in amino
acid sequences and convert them into numeric arrays readable
by neural networks. The goal of the project is to convert amino acid 
sequence data into numeric arrays, and train neural networks on
the resulting arrays. 

Required modules:

numpy
pandas
sklearn
tensorflow
matplotlib

## Project Motivation

### Overview
Classifying proteins is a challenging problem in molecular biology.
Current methods involve isolating and characterizing a protein,
or comparing how its sequence compares to already classified
proteins. However, sequence alignment can fail to identify all
proteins, and relies on a narrow set of sequences being tied to
function. Being able to train a machine learning algorithm
on amino acid information, beyond simple sequence alignment,
could allow for improved classification.

### Problem Statement
Protein sequence data is reported as sequence of one letter
amino acid codes (A, W, V, etc.). These need to be converted
into numeric arrays before they can be used to train NNs.
Different NN architectures need to be evaluated for classification.

### Metrics
Amino acid sequences are converted to numeric
arrays readable by neural networks. Multiple NN architectures
are explored. 

## File Description

models.py - Models trained on the dataset. Includes a 1D CNN, 2D CNN, and 2 LSTMs.

train_model.py - script to train all models in model.py and save
best weights.

seq2mat.py - functions for processing amino acid sequences and returning arrays of data that can be read by machine learning
algorithms

\*.model.best.hdf5 - weight files for various models

model_eval.ipynb - notebook for evaluating models

data/AA_info.csv - amino acid data for transformation

data/AA_keys.csv - processed amino acid data

data/AA_prep.ipynb - notebook for preparing AA_keys.csv

data/Raw_data_prep.ipynb - notebook for cleaning raw protein data

data/Seq_class.csv - cleaned data for training

## Results

The functions for data processing worked well, taking in sequences
composed of one letter amino acid codes and returning matrices
of numeric data suitable for neural networks.

CNNs were not able to classify proteins, likely due to the
computational limitations which forced simpler architectures.

LSTMs showed promise. Even a single, default LSTM from Keras
had reasonable recall for multiple protein classes.

## Acknolwedgements

Data from Kaggle, https://www.kaggle.com/shahir/protein-data-set

DataGenerator code expanded on code provided by Afshine Amidi and Shervine Amidi at https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

