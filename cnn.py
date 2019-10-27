#!/usr/bin/env python
# coding: utf-8
import argparse

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Conv2DTranspose, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('resources'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pyximport
pyximport.install(language_level=3)
from life import *

NROW, NCOL = 20, 20

def generate_samples(delta=1, n=32):
    """
    Generate batch of samples
    
    @return: (end_frames, start_frames)
    """
    batch = np.split(np.random.binomial(1, 0.5, (NROW * n, NCOL)).astype('uint8'), n)
    Yy = [life.make_move(state, 5) for state in batch]
    Xx = [life.make_move(state, 1) for state in Yy]
    Y = np.array([y.ravel() for y in Yy])
    X = np.array([x.ravel() for x in Xx])
    return X, Y
    

def data_generator(delta=1, batch_size=32, ravel=True):
    """
    Can be used along with .fit_generator to generate training samples on the fly
    """
    while True:
        batch = np.split(np.random.binomial(1, 0.5, (NROW * batch_size, NCOL)).astype('uint8'), batch_size)
        Yy = [make_move(state, 5) for state in batch]
        Xx = [make_move(state, delta) for state in Yy]

        if ravel:
            Y = np.array([y.ravel() for y in Yy])
            X = np.array([x.ravel() for x in Xx])
            yield X, Y
        else:
            yield np.array(Xx)[:,:, :, np.newaxis], np.array(Yy)[:, :, :, np.newaxis]



def create_model(n_hidden_convs=2, n_hidden_filters=128, kernel_size=5):
    nn = Sequential()
    nn.add(Conv2D(n_hidden_filters, kernel_size, padding='same', activation='relu', input_shape=(20, 20, 1)))
    nn.add(BatchNormalization())
    for i in range(n_hidden_convs):
        nn.add(Conv2D(n_hidden_filters, kernel_size, padding='same', activation='relu'))
        nn.add(BatchNormalization())
    nn.add(Conv2D(1, kernel_size, padding='same', activation='sigmoid'))
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return nn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dataset', default='resources/train.csv', help='dataset to train for')
    parser.add_argument('test_dataset', default='resources/test.csv', help='dataset to generate predictions for')
    parser.add_argument('--submission-name', '-s', default='submission.csv', help='name of the submission file')

    models = []
    for delta in range(1, 6):
        np.random.seed(42)
        model = create_model(n_hidden_convs=6, n_hidden_filters=256)
        es = EarlyStopping(monitor='loss', patience=9, min_delta=0.001)
        model.fit_generator(data_generator(delta=delta, ravel=False), steps_per_epoch=500, epochs=50, verbose=1, callbacks=[es])
        models.append(model)


    train_df = pd.read_csv('resources/train.csv', index_col=0)
    test_df = pd.read_csv('resources/test.csv', index_col=0)


    submit_df = pd.DataFrame(index=test_df.index, columns=['start.' + str(_) for _ in range(1, 401)])


    for delta in range(1, 6):
        mod = models[delta-1]
        delta_df = test_df[test_df.delta == delta].iloc[:, 1:].values.reshape(-1, 20, 20, 1)
        submit_df[test_df.delta == delta] = mod.predict(delta_df).reshape(-1, 400).round(0).astype('uint8')


    submit_df.to_csv(args.submission_name)

