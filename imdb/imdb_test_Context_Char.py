# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:44:31 2018

@author: jbk48
"""

import os
import tensorflow as tf
import Bi_LSTM
import Imdb_movie
import numpy as np


tf.set_random_seed(0)
## Setting GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
## file name
embedding_filename = "polyglot-en.pkl"   ## pretrained 64 dim word embedding
imdb_filename = "imdb_master.csv"        ## Get imdb movie review dataset
## Get data
movie = Imdb_movie.Movie()
movie.get_embedding(embedding_filename)
label, review = movie.get_data(imdb_filename, "test")
test_X = movie.make_corpus(review)
char_list = movie.char_list(test_X)
movie.get_OOV(char_list)
## Data -> Vector
test_X_ = movie.Convert2Vec(test_X, "Context_Char")
test_Y_ = label

Batch_size = 32
Total_size = len(test_X_)
Vector_size = 64
seq_length = [len(x) for x in test_X_]
Maxseq_length = max(seq_length) 
learning_rate = 0.001
lstm_units = 128
training_epochs = 10
keep_prob = 0.75
num_class = 2

## Placeholders
X = tf.placeholder(tf.float32, shape = [None, Maxseq_length, Vector_size], name = 'X')
Y = tf.placeholder(tf.int32, shape = [None], name = 'Y')
seq_len = tf.placeholder(tf.int32, shape = [None])

BiLSTM = Bi_LSTM.Bi_LSTM(lstm_units, num_class, keep_prob)

with tf.variable_scope("loss", reuse = tf.AUTO_REUSE):
    logits = BiLSTM.logits(X, BiLSTM.W, BiLSTM.b, seq_len)
    loss, optimizer = BiLSTM.model_build(logits, Y, learning_rate)

pred = tf.cast(tf.argmax(logits, 1), tf.int32)
correct_pred = tf.equal(pred, Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

total_batch = int(Total_size / Batch_size)

modelName = "./imdb_Context_Char/Context_Char_classification.ckpt"
saver = tf.train.Saver()


with tf.Session(config = config) as sess:

    sess.run(init)
    saver.restore(sess, modelName)
    print("Model restored")
    
    total_acc = 0
    

    for step in range(total_batch):

        test_batch_X = test_X_[step*Batch_size : step*Batch_size+Batch_size]
        test_batch_Y = test_Y_[step*Batch_size : step*Batch_size+Batch_size]
        batch_seq_length = seq_length[step*Batch_size : step*Batch_size+Batch_size]
            
        test_batch_X = movie.Zero_padding(test_batch_X, Batch_size, Maxseq_length, Vector_size)
            
       
        acc = sess.run(accuracy , feed_dict={X: test_batch_X, Y: test_batch_Y, seq_len: batch_seq_length})
        print("step : {:04d} accuracy= {:.6f}".format(step+1, acc))
        total_acc += acc/total_batch
    
    print("Total Accuracy : {:.6f}".format(total_acc))
    np.savetxt("./imdb_Context_Char/Accuracy.csv", [total_acc], delimiter = ',')
   