# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:39:36 2018

@author: jbk48
"""
import os
import tensorflow as tf
import time
import Bi_LSTM
import Yelp


tf.set_random_seed(0)
## Setting GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
## file name
embedding_filename = "polyglot-en.pkl"   ## pretrained 64 dim word embedding
## Get data
yelp = Yelp.Yelp()
yelp.get_embedding(embedding_filename)
label, review = yelp.get_data("train")
train_X = yelp.make_corpus(review)
char_list = yelp.char_list(train_X)
yelp.get_OOV(char_list)


## Data -> Vector
train_X_ = yelp.Convert2Vec(train_X, "Context_Char")
train_Y_ = label

Batch_size = 32
Total_size = len(train_X)
Vector_size = 64
seq_length = [len(x) for x in train_X]
Maxseq_length = max(seq_length) 
learning_rate = 0.001
lstm_units = 128
training_epochs = 10
keep_prob = 0.75
num_class = 5

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

print("Start training!")

modelName = "./Yelp_Context_Char/Context_Char_classification.ckpt"
saver = tf.train.Saver()


with tf.Session(config = config) as sess:

    start_time = time.time()
    sess.run(init)
    train_writer = tf.summary.FileWriter('./yelp_graph', sess.graph)
    merged = BiLSTM.graph_build()
    
    for epoch in range(training_epochs):

        avg_acc, avg_loss = 0. , 0.
        for step in range(total_batch):

            train_batch_X = train_X_[step*Batch_size : step*Batch_size+Batch_size]
            train_batch_Y = train_Y_[step*Batch_size : step*Batch_size+Batch_size]
            batch_seq_length = seq_length[step*Batch_size : step*Batch_size+Batch_size]
            
            train_batch_X = yelp.Zero_padding(train_batch_X, Batch_size, Maxseq_length, Vector_size)
            
            sess.run(optimizer, feed_dict={X: train_batch_X, Y: train_batch_Y, seq_len: batch_seq_length})
            # Compute average loss
            loss_ = sess.run(loss, feed_dict={X: train_batch_X, Y: train_batch_Y, seq_len: batch_seq_length})
            avg_loss += loss_ / total_batch
            
            acc = sess.run(accuracy , feed_dict={X: train_batch_X, Y: train_batch_Y, seq_len: batch_seq_length})
            avg_acc += acc / total_batch
            print("epoch : {:02d} step : {:04d} loss = {:.6f} accuracy= {:.6f}".format(epoch+1, step+1, loss_, acc))
   
        summary = sess.run(merged, feed_dict = {BiLSTM.loss : avg_loss, BiLSTM.acc : avg_acc})       
        train_writer.add_summary(summary, epoch)
        
    train_writer.close()
    duration = time.time() - start_time
    minute = int(duration / 60)
    second = int(duration) % 60
    print("%dminutes %dseconds" % (minute,second))
    save_path = saver.save(sess, modelName)
    
    print ('save_path',save_path)
    
