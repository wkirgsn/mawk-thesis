import keras

import numpy as np
np.random.seed(2017)
import matplotlib.pyplot as plt
import os
import glob
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")
from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,\
    ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import __version__ as keras_version
from keras.applications.vgg16 import VGG16
from keras.regularizers import l2

Input_param_names = ['ambient',
                     'coolant',
                     'u_d',
                     'u_q',
                     'motor_speed',
                     'torque',
                     'i_d',
                     'i_q']

Target_param_names = ['pm',
                      'stator_yoke',
                      'stator_tooth',
                      'stator_winding']

valset = (31,)
testset = (20,)
loadsets = [4, 6, 10, 11, 20, 27, 29, 30, 31, 32, 36]
file_path = "/home/wilhelmk/Messdaten/PMSM_Lastprofile/hdf/all_load_profiles"
subsequence_len = 100


def load_data(hdf_file):
    train = [pd.read_hdf(hdf_file, key='p' + str(k)) for k in loadsets if k
        not in valset or k not in testset]
    val = [pd.read_hdf(hdf_file, key='p' + str(k)) for k in valset]
    test = [pd.read_hdf(hdf_file, key='p' + str(k)) for k in testset]
    return train, val, test


def get_norm_stats(all):
    temp = all[0].copy()
    for i in range(1, len(all)):
        temp.append(all[i], ignore_index=True)
    return temp.mean(), temp.max(), temp.min()

if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    train_set, val_set, test_set = load_data(file_path)

    all_data = train_set + val_set + test_set

    # normalize data
    s_mean, s_max, s_min = get_norm_stats(all_data)
    all_data_normed = [(a - s_mean) / (s_max - s_min) for a in all_data]

    print(all_data_normed[0].head(10))

    test_set = all_data_normed.pop()
    val_set = all_data_normed.pop()

    val_in = val_set[Input_param_names].values
    val_out = val_set[Target_param_names].values

    test_in = test_set[Input_param_names].values
    test_out = test_set[Target_param_names].values

    input_dim = test_in.shape[1]

    # create model
    model = Sequential()
    model.add(GRU(32, batch_input_shape=(1, 1, input_dim),
                  consume_less='cpu', stateful=True))
    model.add(Dense(4))

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, verbose=0),
    ]

    model.compile(optimizer=sgd, loss='mse')

    """print('in shape: {} out shape: {}'.format(model_in.shape, model_out.shape))
    model.fit(model_in, model_out, batch_size=1, nb_epoch=100,
              validation_data=(val_in, val_out),
              callbacks=callbacks)

    score = model.evaluate(test_in, test_out)
    print(score)"""

    print('Train...')
    for epoch in range(15):
        print('Epoch {}'.format(epoch))
        mean_tr_loss = []
        sample_count = 0
        for sample in all_data_normed:
            sample_count += 1
            print('sample {} of {}'.format(sample_count, len(all_data_normed)))
            # split input and output
            model_in = sample[Input_param_names].values
            model_out = sample[Target_param_names].values

            for i in range(len(sample.index)):
                tr_loss = model.train_on_batch(
                    np.expand_dims(np.expand_dims(model_in[i, :], axis=0),
                                   axis=0), np.atleast_2d(model_out[i, :]))
                mean_tr_loss.append(tr_loss)
            model.reset_states()

            print('loss training = {}'.format(np.mean(mean_tr_loss)))

        mean_te_loss = []
        for i in range(val_in.shape[0]):
            te_loss = model.test_on_batch(
                np.expand_dims(np.expand_dims(val_in[i, :], axis=0), axis=0),
                               np.atleast_2d(val_out[i, :]))
            mean_te_loss.append(te_loss)
        model.reset_states()

        print('loss validation = {}'.format(np.mean(mean_te_loss)))
        print('___________________________________')

    print("Test..")

    y_pred = []
    for j in range(test_in.shape[0]):
        batch = test_in[j, :]
        y_pred.append(model.predict_on_batch(batch[np.newaxis, np.newaxis, :]))
    y_pred = np.vstack(y_pred)
    model.reset_states()


    mean_te_loss = []
    for i in range(test_in.shape[0]):
        batch = test_in[i, :]
        batch_y = test_out[i, :]
        te_loss = model.test_on_batch(batch[np.newaxis, np.newaxis, :],
                           batch_y[np.newaxis, :])
        mean_te_loss.append(te_loss)
    model.reset_states()

    print('loss test = {}'.format(np.mean(mean_te_loss)))
    print('___________________________________')

    time = np.arange(len(test_in.index), dtype=np.float32)
    time /= (2*60)
    plt.plot(time, test_out[:, 0])
    plt.plot(time, y_pred[:, 0])