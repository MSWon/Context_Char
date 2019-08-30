# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 11:21:34 2018

@author: jbk48
"""

import json
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

filename = './yelp_academic_dataset_review.json'

def make_dataset(filename):
    
    data =[]
    print("Reading json file")
    with open(filename, 'rt', encoding = 'UTF-8') as json_file:
        for line in json_file:
            data.append(json.loads(line))
        
    scores = np.array([data[i]['stars'] for i in range(len(data))])
    
    print("Converting to Csv")
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    
    for index in range(1,5+1):
    
        indicies = np.where(scores == index)[0]
        indicies = np.random.choice(indicies, size = 50000, replace = False)
        indicies_train = indicies[0:40000]
        indicies_test = indicies[40000:50000]
      
        corpus_train = pd.DataFrame({"corpus": [data[i]['text'] for i in indicies_train]})
        score_train = pd.DataFrame({"score": [index for j in range(40000)]})  
        df_1 = pd.concat([corpus_train, score_train], axis = 1)
        df_train = pd.concat([df_train,df_1], axis = 0)
        
        corpus_test = pd.DataFrame({"corpus": [data[i]['text'] for i in indicies_test]})
        score_test = pd.DataFrame({"score": [index for i in range(10000)]})  
        df_2 = pd.concat([corpus_test, score_test], axis = 1)
        df_test = pd.concat([df_test,df_2], axis = 0)
        
    df_train = shuffle(df_train)
    df_test = shuffle(df_test)
    
    df_train.to_csv("yelp_train.csv" , sep = "," , index = False)
    df_test.to_csv("yelp_test.csv" , sep = "," , index = False)
    print("Done!")


make_dataset(filename)

