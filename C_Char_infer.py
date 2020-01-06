# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:25:37 2018

@author: jbk48
"""
## added 2
import numpy as np
import tensorflow as tf
import Char2BiLSTM
import Context_Char
from keras.preprocessing.sequence import pad_sequences


class OOV_infer():
    
    def __init__(self, char_list):
       
        ## Parameters
        char_max_len = 100
        lstm_units = 100
        wordemb_size = 100
        keep_prob = 1.0
        

        modelName = "./C_Char/CC_model.ckpt"
        char_emb_name = "./C_Char/embedding.npy"
            

        ## Place holders
        self.char_length = tf.placeholder(tf.int32, shape = [None])
        self.context_vec = tf.placeholder(tf.float32, shape = [None, lstm_units])
        self.word_label = tf.placeholder(tf.float32, shape = [None, wordemb_size])
        self.input_index = tf.placeholder(tf.int32, shape = [None, char_max_len])
              
        self.c2vec = Char2BiLSTM.Char2Vec()
        self.c2vec.get_char_list(char_list)
        self.char_embedding = self.c2vec.load_char_embedding(char_emb_name)
        self.embed = tf.nn.embedding_lookup(self.char_embedding, self.input_index)  ## lookup table for char embedding
        self.context_char = Context_Char.Context()
        self.context_char.get_embedding()   ## Get pretrained word embedding
        
        c2b = Char2BiLSTM.Char2BiLSTM(lstm_units, keep_prob)
        context_vec_ = tf.nn.l2_normalize(self.context_vec, axis = 1)
        initial_state = tf.contrib.rnn.LSTMStateTuple(h=context_vec_, c=context_vec_)
        final_state = c2b.BiLSTM_layer(self.embed, initial_state, self.char_length)
        self.logits = c2b.MLP_layer(final_state = final_state, wordembedding_size = wordemb_size)
        
        self.vocabulary = self.context_char.vocabulary
        
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        saver.restore(self.sess, modelName)
    
    def make1D(self, data):
        result = []
        for x in data:
            result += x
        return result
    
    def mean_vec(self, context):
        return np.mean([self.context_char.index2vec[self.vocabulary[word]] for word in context], axis=0)
    
    def OOV(self, word, corpus, window = 5):  ## corpus = clean_X
        
        
        char_len = len(word)
        char = [[self.c2vec.char_dict[char] for char in word]]
        pad_char = pad_sequences(char, maxlen = self.c2vec.max_len, padding = "post")
        
        context_list = []
        
        for sent in corpus:
            
            sent2 = np.array(sent)
            indicies = np.where(sent2 == word)[0]        
            context = []
            
            for index in indicies:
                
                if(index >= window):
                    left_context = sent[index-window:index]
                elif(index < window and index > 0):
                    left_context = sent[0:index]
                else:
                    left_context = []
                
                if(index+window >= len(sent)):
                    if(index == len(sent)-1):
                        right_context = []
                    else:
                        right_context = sent[index+1:len(sent)]
                elif(index+window < len(sent)):
                    right_context = sent[index+1:index+window+1]
                
                context += (left_context + right_context)

            for token in context:
                if not token in self.vocabulary:
                    if(token.lower() in self.vocabulary):
                        context[context.index(token)] = token.lower()
                    else:
                        context[context.index(token)] = "<UNK>" 
                    
            if(len(context)>0):    
                context_list.append(context)  
            
        context_list = self.make1D(context_list)
                                                    
        infer_context_vec = np.reshape(self.mean_vec(context_list), (-1,100))
        
        feed_dict = {self.input_index:pad_char, self.char_length:[char_len], 
                     self.context_vec:infer_context_vec}
        pred = self.sess.run(self.logits , feed_dict = feed_dict)
        
        return pred.flatten()
    
    def OOV2(self, word, sent, window = 5):
        
        
        char_len = len(word)
        char = [[self.c2vec.char_dict[char] for char in word]]
        pad_char = pad_sequences(char, maxlen = self.c2vec.max_len, padding = "post")
        sent2 = np.array(sent)
        indicies = np.where(sent2 == word)[0]
        
        context = []
        
        for index in indicies:
            
            if(index >= window):
                left_context = sent[index-window:index]
            elif(index < window and index > 0):
                left_context = sent[0:index]
            else:
                left_context = []
            
            if(index+window >= len(sent)):
                if(index == len(sent)-1):
                    right_context = []
                else:
                    right_context = sent[index+1:len(sent)]
            elif(index+window < len(sent)):
                right_context = sent[index+1:index+window+1]
            
            context += (left_context + right_context)
            
        for word in context:
            if not word in self.vocabulary:
                if(word.lower() in self.vocabulary):
                    context[context.index(word)] = word.lower()
                else:
                    context[context.index(word)] = "<UNK>"
                                
        infer_context_vec = np.reshape(self.mean_vec(context), (-1,100))
        
        feed_dict = {self.input_index:pad_char, self.char_length:[char_len], 
                     self.context_vec:infer_context_vec}
        pred = self.sess.run(self.logits , feed_dict = feed_dict)
        return pred.flatten()

    def average(self, word, sent, window = 5):
        
        
        sent2 = np.array(sent)
        indicies = np.where(sent2 == word)[0]
        
        context = []
        
        for index in indicies:
            
            if(index >= window):
                left_context = sent[index-window:index]
            elif(index < window and index > 0):
                left_context = sent[0:index]
            else:
                left_context = []
            
            if(index+window >= len(sent)):
                if(index == len(sent)):
                    right_context = []
                else:
                    right_context = sent[index+1:len(sent)]
            elif(index+window < len(sent)):
                right_context = sent[index+1:index+window+1]
            
            context += (left_context + right_context)
            
        for word in context:
            if not word in self.vocabulary:
                context[context.index(word)] = "<UNK>"
                                
        infer_context_vec = self.mean_vec(context)

        return infer_context_vec


