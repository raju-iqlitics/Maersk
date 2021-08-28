# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 12:05:42 2021

@author: A1019089
"""

import numpy as np
import pandas as pd
import re # regular expression is used to find a specific text or letter/ word in a sentence or paragraph
import nltk
#nltk.download('stopwords')
import string
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost
import pickle

import os

path = os.getcwd()

stemmer = PorterStemmer()

news_df = pd.read_csv(path+'\train\train.csv')

print(news_df.shape)


#checking null values in column

news_df.isnull().sum()

#we found below result
#
#id           0
#title      558
#author    1957
#text        39
#label        0

## instead of droping missing text i.e 39 here i am adding title and author to text

news_df = news_df.fillna('')

news_df['content'] = news_df['author'] + " " + news_df['title']



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

news_df = preprocess(news_df)


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


news_df['content'] = news_df['content'].apply(stemming)
print(news_df['content'])

Xtrain = news_df['content'].values
ytrain = news_df['label'].values

vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words='english')
X = vectorizer.fit_transform(Xtrain)

pickle.dump(vectorizer, open(path+'\model\vectorizer.pickle", "wb"))

classifier = xgboost.XGBClassifier(max_depth=16,booster='gbtree',verbose= 1,learning_rate=0.02,n_estimators=500,min_child_weight=5 )

classifier.fit(X, ytrain)

pickle.dump(classifier, open(path+'\model\classifier.pickle", "wb"))

### train accuracy

y_pred_train = classifier.predict(X)
accuracy_train = accuracy_score(ytrain,y_pred_train)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,X = X,y= ytrain , cv = 10)


print("-------------------------------")
print("Accuracy score on training data: ", accuracy_train)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f}".format(accuracies.std()*100))

#-------------------------------
#Accuracy score on training data:  0.9840865384615385
#Accuracy: 97.77 %
#Standard Deviation: 0.26