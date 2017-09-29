# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 13:40:35 2017

@author: hanjoh
"""





import xlrd
import numpy as np
from sklearn.cross_validation import *
from data_input import improt_data
from sklearn.feature_extraction.text import CountVectorizer
import xlrd
#import bayesian_classifier as bc
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline

TRAIN_SET_LOC = './training_dataset.txt'
TEST_SET_LOC = './test_dataset.xlsx'
X,y = improt_data()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4,random_state=23)


#this calculates sentence sentiments using NLTK Vader, a Rule Based classifier 
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

result_nltk = []
sid = SentimentIntensityAnalyzer()
for i in X_test:
    ss = sid.polarity_scores(i)
    if ss["pos"] > ss["neg"]:
        result_nltk.append(1)
    else:
        result_nltk.append(0)

true_label =np.array(list(map(int,y_test)))
result = np.array(result_nltk)
acc = np.mean(result == true_label) * 100.
print("Testing Accuracy, Vader %.2f%%" % acc)


'''
A Pipeline 
CountVectorizer: convert a vector into feature vectors, N by K, N is the total 
number of samples, and K is # of features

TfidfTransformer: apply tf-idf to the input vector, tf-idf will reconsider how 
important is a word based on the apperance in other samples

BernoulliNB: assume the input features from Bernouli distributions,
 then apply Navie Bayes rule to the product of all likelihoods to obtain posterios

'''
#build a pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', BernoulliNB()),]) 
text_clf.fit(X_train,y_train)

#training acc
result_sk = text_clf.predict(X_train)

true_label =np.array(list(map(int,y_train)))
result_sk = np.array(list(map(int,result_sk)))
acc = np.mean(result_sk == true_label) *100.
print("Training Accuracy, Bernoulli Naive Bayes with tf-idf: %.2f%%" % acc)


#testing acc
result_sk = text_clf.predict(X_test)


true_label =np.array(list(map(int,y_test)))
result_sk = np.array(list(map(int,result_sk)))
acc = np.mean(result_sk == true_label) *100.
print("Testing Accuracy, Bernoulli Naive Bayes with tf-idf: %.2f%%" % acc)


