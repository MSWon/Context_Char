# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 21:30:00 2018

@author: jbk48
"""
import os
import time
import pandas as pd
import tensorflow as tf
import Char2BiLSTM
import Context_Char
from six.moves import xrange

## Setting GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
## file name
embedding_filename = "polyglot-en.pkl"   ## pretrained 64 dim word embedding
df = pd.read_csv("imdb_master.csv", encoding = 'cp1252')
df_train = df[df['type'] == 'train']
## df_train = df_train[df_train['label'] != 'unsup']
reviews = list(df_train['review'])
## Parameters
window = 5
batch_size = 32
char_max_len = 100
char_emb_size = 20
lstm_units = 64
wordemb_size = 64
keep_prob = 0.75
training_epochs = 10

## Place holders
char_length = tf.placeholder(tf.int32, shape = [None])
context_vec = tf.placeholder(tf.float32, shape = [None, lstm_units])
word_label = tf.placeholder(tf.float32, shape = [None, wordemb_size])
input_index = tf.placeholder(tf.int32, shape = [None, char_max_len])

c2vec = Char2BiLSTM.Char2Vec(char_max_len)
c2vec.initialize_char_embedding(char_emb_size) ## Initialize char embedding
context_char = Context_Char.Context()
corpus = context_char.make_corpus(reviews)   ## make training data corpus
context_char.get_embedding(embedding_filename)   ## Get pretrained word embedding
embed = tf.nn.embedding_lookup(c2vec.char_embedding, input_index)  ## lookup table for char embedding

c2b = Char2BiLSTM.Char2BiLSTM(lstm_units, keep_prob)
initial_state = tf.contrib.rnn.LSTMStateTuple(h=context_vec, c=context_vec)
final_state = c2b.BiLSTM_layer(embed, initial_state, char_length)
logits = c2b.MLP_layer(final_state = final_state, wordembedding_size = wordemb_size)
loss, optimizer = c2b.model_build(logits, word_label)



total_batch = int(len(corpus)/batch_size)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
modelName = "./CC_model.ckpt"


with tf.Session(config = config) as sess:
    
    sess.run(init)
    start_time = time.time()
    
    for epoch in xrange(training_epochs):
        
        context_char.initialize_index()  ## initialize index to 0
        epoch_loss = 0.
        
        for step in xrange(total_batch):
            
            batch_target, batch_context = context_char.build_index(corpus, window, context_char.vocabulary, batch_size)
            batch_context_vec = context_char.Context_vector(context_char.index2vec, batch_context)
            char_len, pad_char = c2vec.char2index(batch_target, context_char.reverse_vocabulary) 
            step_loss = 0.
            
            for i in xrange(batch_size):
                
                label_X = c2b.Convert_labels(batch_target[i], context_char.index2vec)
                feed_dict = {input_index:pad_char[i],char_length:char_len[i], 
                             context_vec:batch_context_vec[i], word_label:label_X}
                _, loss_val = sess.run([optimizer, loss],  feed_dict = feed_dict)            
                step_loss += loss_val / batch_size
            
            epoch_loss += step_loss / total_batch
                
            print("epoch:{:02d}, step:{:03d}, loss : {:.6f}".format(epoch+1,step+1,step_loss))
            save_path = saver.save(sess, modelName)
        
        print("#### epoch:{:02d}, avg_loss : {:.6f} ####".format(epoch+1, epoch_loss))
              
    duration = time.time() - start_time
    minute = int(duration / 60)
    second = int(duration) % 60
    print("%dminutes %dseconds" % (minute,second))
