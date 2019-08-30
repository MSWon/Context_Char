# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 21:53:34 2018

@author: jbk48
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 20:22:39 2018

@author: jbk48
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import numpy as np
import pandas as pd
import ag_news
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

max_cap = 100

news = ag_news.ag_news()
metrics = ag_news.Metrics()
train_X, train_Y = news.read_data("./train.csv")
test_X, test_Y = news.read_data("./test.csv")
train_X, train_Y = news.clean_text(train_X, train_Y)
test_X, test_Y = news.clean_text(test_X, test_Y)

X  = np.concatenate((train_X ,test_X), axis = 0)
Y = np.concatenate((train_Y ,test_Y), axis = 0)
# Generate a cleaned reviews array from original review texts

sentences = [' '.join(r) for r in X]
char_list = news.char_list(X)
news.get_infer(char_list)
news.fit_text(sentences)
embeddings_matrix = news.emb_matrix(X ,"Context_Char")

X = pad_sequences(news.sequences, maxlen=max_cap, padding='pre')


train_cap = len(train_X)

X_train, Y_train = X[:train_cap], Y[:train_cap]
X_test, Y_test = X[train_cap:], Y[train_cap:]

#Re-generate reviews_encoded, X, and Y after changing max_cap

model = Sequential()
model.add(Embedding(len(news.word_index)+1, 100, input_length=max_cap, weights=[embeddings_matrix], trainable=False))
model.add(LSTM(60, return_sequences=True, recurrent_dropout=0.5))
model.add(Dropout(0.5))
model.add(LSTM(60, recurrent_dropout=0.5))
model.add(Dropout(0.5))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
# fit model
model.fit(X_train, Y_train, batch_size=64, epochs=20, validation_data=(X_test, Y_test), callbacks = [metrics])
print("random")
