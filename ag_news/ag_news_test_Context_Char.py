# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 23:09:13 2018

@author: jbk48
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 22:56:52 2018

@author: jbk48
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:33:26 2018

@author: jbk48
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:39:36 2018

@author: jbk48
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorflow as tf
tf.set_random_seed(0)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
import Bi_LSTM
import numpy as np
import ag_news
import Context_Char

news = ag_news.ag_news()
test_X, test_Y = news.read_data("./test.csv")
test_X, test_Y = news.clean_text(test_X, test_Y)
test_Y = [np.argmax(y) for y in test_Y]


context_char = Context_Char.Context()
context_char.get_embedding()   ## Get pretrained word embedding
char_vocab = list(context_char.vocabulary)
char_list = context_char.char_list(char_vocab)   ## make unique char list
char_list = sorted(char_list)
news.get_infer(char_list, "large_book")

## Data -> Vector
test_X_ = news.Convert2Vec(test_X, "Context_Char")
test_Y_ = test_Y

Batch_size = 128
Total_size = len(test_X)
Vector_size = 100
seq_length = [len(x) for x in test_X]
Maxseq_length = max(seq_length) 
learning_rate = 0.001
lstm_units = 128
training_epochs = 1
keep_prob = 1.0
num_class = 4

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


modelName = "./ag_news_Context_Char/Context_Char_classification.ckpt"
saver = tf.train.Saver()


with tf.Session(config = config) as sess:

    sess.run(init)
    saver.restore(sess, modelName)
    print("model restored")
    for epoch in range(training_epochs):

        avg_acc, avg_loss = 0. , 0.
        for step in range(total_batch):

            train_batch_X = test_X_[step*Batch_size : step*Batch_size+Batch_size]
            train_batch_Y = test_Y_[step*Batch_size : step*Batch_size+Batch_size]
            batch_seq_length = seq_length[step*Batch_size : step*Batch_size+Batch_size]
            
            train_batch_X = news.Zero_padding(train_batch_X, Batch_size, Maxseq_length, Vector_size)

            # Compute average loss
            loss_ = sess.run(loss, feed_dict={X: train_batch_X, Y: train_batch_Y, seq_len: batch_seq_length})
            avg_loss += loss_ / total_batch
            
            acc = sess.run(accuracy , feed_dict={X: train_batch_X, Y: train_batch_Y, seq_len: batch_seq_length})
            avg_acc += acc / total_batch

        print("Loss = {:.6f} Accuracy = {:.6f}".format(avg_loss, avg_acc))
    
