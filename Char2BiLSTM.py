# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 10:45:33 2018

@author: jbk48
"""
## update

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

        
class Char2Vec():
    
    def __init__(self, input_char, max_len=100):
        
        self.max_len = max_len
        self.seq_len = tf.placeholder(tf.int32, shape = [None])
        self.char_list = input_char
        self.char_dict = {char:index for index, char in enumerate(self.char_list)}

        
    def char2index(self, batch_target, reverse_vocabulary):

        char_len_list = []
        pad_char_list = []
        
        for batch in batch_target:
            char_len_sub = []
            char_sub = []
            for index in batch:
                word = reverse_vocabulary[index]
                char_len = len(word)
                char_len_sub.append(char_len)
                
                char_list = [self.char_dict[char] for char in word]   
                char_sub.append(char_list)
                
            pad_char_sub = pad_sequences(char_sub, maxlen = self.max_len, padding = "post")
            char_len_list.append(char_len_sub)
            pad_char_list.append(pad_char_sub)
            
        return char_len_list, pad_char_list
    
    def initialize_char_embedding(self, embedding_size, vocab_size):
        
        self.char_embedding = tf.Variable(
                tf.random_normal([vocab_size, embedding_size], mean = 0.0, stddev = 1.0), name = "embedding")   
    
    def get_embedding(self, path):
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, path)
        char_embedding = sess.graph.get_tensor_by_name("embedding:0")
        sess.close()
        return char_embedding
    
    def char_embedding(self, char_list):
        
        embed = tf.nn.embedding_lookup(self.char_embedding, char_list)
        return embed
    
    
                
class Char2BiLSTM():
    
    def __init__(self, lstm_units=64, keep_prob=0.75):
        
        self.lstm_units = lstm_units
        self.keep_prob = keep_prob
        
        with tf.variable_scope('forward', reuse = tf.AUTO_REUSE):
            self.lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_units, forget_bias=1.0, state_is_tuple=True)
            self.lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_fw_cell, output_keep_prob = keep_prob)
            
        with tf.variable_scope('backward', reuse = tf.AUTO_REUSE):
            self.lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_units, forget_bias=1.0, state_is_tuple=True)
            self.lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_bw_cell, output_keep_prob = keep_prob)
    
    def BiLSTM_layer(self, X, input_state, seq_len):
        ## input_state shape : (2, Batch_size, lstm_unit)
        (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(self.lstm_fw_cell, 
                                            self.lstm_bw_cell,dtype=tf.float32, initial_state_fw=input_state, 
                                            initial_state_bw = input_state, inputs=X, sequence_length=seq_len)
        ## concat fw, bw final states
        outputs = tf.concat([states[0][1], states[1][1]], axis=1) ## shape (?,lstm_unit*2)
        return(outputs)
        
    def MLP_layer(self, final_state, wordembedding_size=64):
        
        with tf.variable_scope('MLP_Weights', reuse = tf.AUTO_REUSE):
            
            self.W1 = tf.get_variable(name="W1", shape=[2 * self.lstm_units, wordembedding_size],
                                     dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            self.b1 = tf.get_variable(name="b1", shape=[wordembedding_size], dtype=tf.float32,
                                     initializer=tf.zeros_initializer())      
        
        logits = tf.matmul(final_state, self.W1) + self.b1
        return logits

    def Convert_labels(self, labels, index2vec):
        return [index2vec[index] for index in labels]
    
    def model_build(self, logits, labels, learning_rate = 0.001):

        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(tf.square(logits-labels))  ## l2 loss
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss) # Adam Optimizer
            
        return loss, optimizer
    
    def graph_build(self):
        
        self.loss = tf.placeholder(tf.float32)
        tf.summary.scalar('Loss', self.loss)
        merged = tf.summary.merge_all()
        return merged
    
        

    
    

