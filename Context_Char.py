# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 20:59:40 2018

@author: jbk48
"""

import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk import tokenize


class Context():
    
    def __init__(self):
        self.data_index = 0
        
    def read_data(self, filename): 
        with open(filename, 'r',encoding='utf-8') as f:
            data = [tokenize.word_tokenize(line) for line in f.read().splitlines()]
        return data
    
    def make_corpus(self, corpus):
        return [tokenize.word_tokenize(sent) for sent in corpus]
    
    def get_embedding(self, filename):
    
        words, vector = pd.read_pickle(filename)  ## polyglot-en.pkl   
        self.vocabulary = {word:index for index,word in enumerate(words)}
        self.reverse_vocabulary = dict(zip(self.vocabulary.values(), self.vocabulary.keys()))
        self.index2vec = vector
    
    def initialize_index(self):
        self.data_index = 0
        
    def build_index(self, corpus, window, vocabulary, batch_size):
    
        def replace(corpus):
            for sent in corpus:
                for word in sent:
                    if not word in vocabulary:
                        sent[sent.index(word)] = "<UNK>"
            return corpus
        
        corpus = replace(corpus)
        
        context_list = []
        target_list = []    
        batch_corpus = corpus[self.data_index:self.data_index+batch_size]
        
        for sent in batch_corpus:
            context_sub = []
            target_sub = []
            for i in range(len(sent)):
                if(sent[i] == "<UNK>" or len(sent[i]) <= 2):   ## PASS if target is <UNK> or shorter than 2
                    next
                else:
                    target = vocabulary[sent[i]]
                    context = [vocabulary[sent[i+j]] for j in range(-window,window+1)
                                 if i+j >= 0 and not j==0 and i+j < len(sent)]            
                    context_sub.append(context)
                    target_sub.append(target)
                
            context_list.append(context_sub)
            target_list.append(target_sub)
                        
        self.data_index += batch_size   
        return target_list, context_list
    
    ## target, context = build_index(corpus2, 5, vocabulary, 2)
    
    def Context_vector(self, index2vec, context):
    
        batch_avg_vec = []
        for batch in context:
            avg_vec_list = [np.mean(index2vec[index_list], axis=0) for index_list in batch]
            batch_avg_vec.append(avg_vec_list)
          
        return batch_avg_vec


