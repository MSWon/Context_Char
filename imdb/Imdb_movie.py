# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:10:56 2018

@author: jbk48
"""

import pandas as pd
import numpy as np
import C_Char_infer 
import re
from nltk import tokenize
from nltk.corpus import stopwords
from itertools import chain

class Movie():
    
    def __init__(self):
               
        self.stop = set(stopwords.words('english'))
        ## self.stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}','<br />'])
    
    def get_OOV(self, char_list):
        self.infer = C_Char_infer.OOV_infer(char_list) 
    
    def get_embedding(self, filename):
    
        words, vector = pd.read_pickle(filename)  ## polyglot-en.pkl   
        self.vocabulary = {word:index for index,word in enumerate(words)}
        self.reverse_vocabulary = dict(zip(self.vocabulary.values(), self.vocabulary.keys()))
        self.index2vec = vector
        
    def get_data(self, filename, istrain = "train"):
        df = pd.read_csv(filename, encoding = 'cp1252')
        df_train = df[df['type'] == istrain]
        df_train = df_train[df_train['label'] != 'unsup']
        df_train = df_train.sample(frac=1)
        
        label = []
        for l in df_train['label']:
            if l == "neg":
                label.append(0)
            else:
                label.append(1)
        
        reviews = list(df_train['review'])
        return label, reviews

    def make_corpus(self, corpus):
        print("Making corpus!\n Could take few minutes!")
        tokens = []
        for sent in corpus:
            sent = re.sub("<br />", "", sent)
            t = [token for token in tokenize.word_tokenize(sent) if token not in self.stop]
            if(len(t) > 400):
                t = t[0:400]
            tokens.append(t)
        print("Tokenize done!")
        return tokens   
    
    def char_list(self, tokens):
        t = np.array(tokens).flatten()
        s = [list(set(chain.from_iterable(elements))) for elements in t]
        s = np.array(s).flatten()
        char = list(set(chain.from_iterable(s)))       
        c = pd.DataFrame(char)
        c.to_csv("./char_list.csv", sep = ",")
        print("char_list saved!")
        return char
    
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
                        sub.append(np.random.uniform(-0.25,0.25,64)) ## Random embedding
                    elif(unk == "UNK"):
                        unk_index = self.vocabulary["<UNK>"]
                        sub.append(self.index2vec[unk_index])
                    elif(unk =="Context_Char"):                        
                        sub.append(self.infer.OOV(word, sent))
        
            word_vec.append(sub)
        return word_vec
        
    def Zero_padding(self, train_batch_X, Batch_size, Maxseq_length, Vector_size):
        zero_pad = np.zeros((Batch_size, Maxseq_length, Vector_size))
        for i in range(Batch_size):
            zero_pad[i,:np.shape(train_batch_X[i])[0],:np.shape(train_batch_X[i])[1]] = train_batch_X[i]
                
        return zero_pad
                
                
    

