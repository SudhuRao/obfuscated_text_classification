# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:57:03 2020

@author: sudhanva
"""

from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.layers import MaxPooling1D, Conv1D,BatchNormalization

from config import cfg


#RNN structure
def RNN():
    inputs = Input(name='inputs',shape=[cfg.max_len])
    layer = Embedding(input_dim    = cfg.max_vocab,
                      output_dim   = 13,
                      input_length = cfg.max_len)(inputs)
    layer = BatchNormalization(momentum=0.8)(layer)
    layer = Dropout(0.2)(layer)
    layer = Conv1D(filters=64,
                     kernel_size=5,
                     padding='valid',
                     activation='relu',
                     strides=1)(layer)

    layer = MaxPooling1D(pool_size=4)(layer)
    layer = LSTM(64)(layer)
    layer = BatchNormalization(momentum=0.8)(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(128,name='FC1')(layer)    
    layer = Dense(12,name='out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    
    return model

