# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 14:56:13 2021

@author: A1019089
"""

import pandas as pd
import pickle
import string
import re 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import os

path = os.getcwd()
stemmer = PorterStemmer()

news_dataset_test = pd.read_csv(path+'\test\test.csv')


news_dataset_test.isnull().sum()

news_dataset_test = news_dataset_test.fillna('')

news_dataset_test['content'] = news_dataset_test['author'] + " " + news_dataset_test['title']


## preprocessing steps
def preprocess(data):
    
    # take only char those in ascii and remove punctuation
    data["content"] = data["content"].tolist()
    data["content"] = data['content'].str.replace('\d+', '')
    data["content"] = data['content'].replace('\s+', ' ', regex=True)
    data["content"]=  data["content"].str.encode('ascii', 'ignore').str.decode('ascii')
    data["content"] = data["content"].apply(lambda x:''.join([i for i in x 
                                                  if i not in string.punctuation]))
    
    
    
    return data

news_dataset_test = preprocess(news_dataset_test)


def stemming(content):
    
    # this basically replaces everything other than lower a-z & upper A-Z with a ' '
    stemmed_content = re.sub('[^a-zA-Z]', ' ',content) 
    # to make all text lower case
    stemmed_content = stemmed_content.lower() 
    # this basically splits the line into words with delimiter as ' '
    stemmed_content = stemmed_content.split() 
    
     # basically remove all the stopwords and apply stemming to the final data
    stemmed_content = [stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    
    # this basically joins back and returns the cleaned sentence
    stemmed_content = ' '.join(stemmed_content) 
    return stemmed_content

news_dataset_test['content'] = news_dataset_test['content'].apply(stemming)


Xtest = news_dataset_test['content'].values

vectorizer = pickle.load(open(path+'\model\vectorizer.pickle", 'rb'))
X = vectorizer.transform(Xtest)


classifier = pickle.load(open(path+'\model\classifier.pickle",'rb'))

y_pred_test = classifier.predict(X)

news_dataset_test['label'] =  y_pred_test

# Deleting content column
del news_dataset_test['content']

news_dataset_test.to_csv(path+'\model\outputTest.csv', index =False)

print(len(y_pred_test))



