# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 21:13:33 2018

@author: jbk48
"""

import pandas as pd
import numpy as np
import re
import C_Char_infer
import mimick_infer
from sklearn.utils import shuffle
from nltk import tokenize
from itertools import chain
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.callbacks import Callback
from nltk.corpus import stopwords

class ag_news():
    
    def __init__(self):
        self.stop = stopwords.words('english')
    def get_infer(self, char_list):
        self.infer = C_Char_infer.OOV_infer(char_list)
        ## self.embeddings_index = self.infer.context_char.embeddings_index
        
    def mimick(self, char_list):
        self.m_infer = mimick_infer.mimick_infer(char_list)    
        
    def read_data(self, filename):
        
        ## "./ag_news_csv/train.csv"
        data = pd.read_csv(filename)
        data = shuffle(data)       
        labels = data.iloc[:,0]
        title = data.iloc[:,2]       
        encoder = LabelBinarizer()
        encoder.fit(labels)
        labels = encoder.transform(labels)       
        return title, labels
    
    def clean_text(self, corpus, labels):
        print("Making corpus!\n Could take few minutes!")               
        tokens = []
        index_list = []
        index = 0
        for sent in corpus:
            text = re.sub('<br />', ' ', sent)
            text = re.sub('[^a-zA-Z]', ' ', sent)
            t = [token for token in tokenize.word_tokenize(text) if not token in self.stop]

            if(len(t) > 200):
                t = t[0:200]
            if(len(t) > 10):
                tokens.append(t)
                index_list.append(index)
            index += 1
            
        labels = labels[index_list]
        print("Tokenize done!")
        return tokens, labels

    def char_list(self,tokens):
        
        t = np.array(tokens).flatten()
        s = [list(set(chain.from_iterable(elements))) for elements in t]
        s = np.array(s).flatten()
        char = list(set(chain.from_iterable(s)))       
        c = pd.DataFrame(char)
        c.to_csv("./char_list.csv", sep = ",")
        print("char_list saved!")
        return char

    def get_embedding(self, filename = "polyglot-en.pkl"):
        print("Getting polyglot embeddings!")
        words, vector = pd.read_pickle(filename)  ## polyglot-en.pkl   
        self.vocabulary = {word:index for index,word in enumerate(words)}
        self.reverse_vocabulary = dict(zip(self.vocabulary.values(), self.vocabulary.keys()))
        self.index2vec = vector
    
    def fit_text(self, sentences):
        
        tokenizer = Tokenizer(lower=True)
        tokenizer.fit_on_texts(sentences)
        self.sequences = tokenizer.texts_to_sequences(sentences)
        self.word_index = tokenizer.word_index

    def emb_matrix(self, corpus, oov = "random"):
        
        self.vocab_size = len(self.word_index)
        self.embeddings_matrix = np.zeros((self.vocab_size+1, 100))
        
        self.oov_count = 0
        
        for word, i in self.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
                       
            if word in self.embeddings_index:
                embedding_vector = self.embeddings_index.get(word)
                self.embeddings_matrix[i] = embedding_vector
            else:
                if(oov == "unk"):
                    self.embeddings_matrix[i] = self.embeddings_index['unk']
                elif(oov == "random"):
                    self.embeddings_matrix[i] = np.random.uniform(-1,1,100) 
                elif(oov == "Context_Char"):
                    self.embeddings_matrix[i] = self.infer.OOV(word,corpus)
                
                self.oov_count += 1
        
        return self.embeddings_matrix
    
    def Convert2Vec(self, doc, unk = "Random"):
        word_vec = []
        for sent in doc:
            sub = []
            for word in sent:
                if(word in self.vocabulary):
                    index = self.vocabulary[word]
                    sub.append(self.index2vec[index])
                else:
                    if(unk == "Random"):
                        sub.append(np.random.normal(0,1,64)) ## Random embedding
                    elif(unk == "UNK"):
                        unk_index = self.vocabulary["<UNK>"]
                        sub.append(self.index2vec[unk_index])
                    elif(unk =="Context_Char"):                        
                        sub.append(self.infer.OOV2(word,sent))
                    elif(unk == "average"):
                        sub.append(self.infer.average(word, sent))
                    elif(unk == "mimick"):
                        sub.append(self.m_infer.mimick(word))
        
            word_vec.append(sub)
        return word_vec
        
    def Zero_padding(self, train_batch_X, Batch_size, Maxseq_length, Vector_size):
        zero_pad = np.zeros((Batch_size, Maxseq_length, Vector_size))
        for i in range(Batch_size):
            zero_pad[i,:np.shape(train_batch_X[i])[0],:np.shape(train_batch_X[i])[1]] = train_batch_X[i]
                
        return zero_pad
    
class Metrics(Callback):
    
    def on_train_begin(self, logs={}):
        
        self.val_f1s = []
        self.val_accuracy = []
     
    def on_epoch_end(self, epoch, logs={}):
        
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict,average='micro')
        _val_accuracy = accuracy_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_accuracy.append(_val_accuracy)

        print("val_f1: {}".format(_val_f1))
        print("val_accuracy: {}".format(_val_accuracy))
        return
            
        
