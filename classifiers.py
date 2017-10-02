# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 11:35:29 2017

@author: Johnny
"""

from data_input import get_sparse_data,get_count_sparse_data
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
#import scipy

def check_acc(trained_clf,clf_name,val_acc = False):
    #print training and testing acc automatically 
    #training acc
    result_sk = trained_clf.predict(X_train_sparse)
    
    true_label =np.array(list(map(int,y_train)))
    result_sk = np.array(list(map(int,result_sk)))
    acc = np.mean(result_sk == true_label) *100.
    print("Training Accuracy, %s with tf-idf: %.2f%%" % (clf_name,acc))
    
    #val acc
    if val_acc:
        result_sk = trained_clf.predict(X_val_sparse)
    
        true_label =np.array(list(map(int,y_val)))
        result_sk = np.array(list(map(int,result_sk)))
        acc = np.mean(result_sk == true_label) *100.
        print("Validation Accuracy, %s with tf-idf: %.2f%%" % (clf_name,acc))
    
    
    #testing acc
    result_sk = trained_clf.predict(X_test_sparse)
    
    
    true_label =np.array(list(map(int,y_test)))
    result_sk = np.array(list(map(int,result_sk)))
    acc = np.mean(result_sk == true_label) *100.
    print("Testing Accuracy, %s with tf-idf: %.2f%%" % (clf_name,acc))
    
    precision, recall, _,_ = precision_recall_fscore_support(true_label,result_sk,labels=[0,1,2])
    print("natural precision %.4f, natural recall %.4f" % (precision[0], recall[0]))
    print("pos precision %.4f, pos recall %.4f" % (precision[1], recall[1]))
    print("neg precision %.4f, neg recall %.4f\n" % (precision[2], recall[2]))

if __name__ == "__main__":
    [y_train,X_raw_train,X_train_sparse],[y_val,X_raw_val,X_val_sparse],[y_test,X_raw_test,X_test_sparse] = get_sparse_data(max_df= 0.992,min_df=0.0001)
    
    text_clf = BernoulliNB()
    text_clf.fit(X_train_sparse,y_train)
    check_acc(text_clf,"BernoulliNB")
    
    text_clf = MultinomialNB()
    text_clf.fit(X_train_sparse,y_train)
    check_acc(text_clf,"MultinomialNB")
    
    text_clf = SGDClassifier(loss ='hinge',
                  penalty = 'l1',
                  alpha = 0.00018,
                  n_iter = 100,
                  shuffle = True)
    text_clf.fit(X_train_sparse,y_train)
    check_acc(text_clf,"SGD",True)
    
    text_clf = SVC(C=30, kernel='rbf', degree=2, gamma=100)
    text_clf.fit(X_train_sparse,y_train)
    check_acc(text_clf,"rbf",True)
    
#    parameters = {
##                   'C':scipy.stats.expon(scale=100),
#    #               'degree':(2,3,4),
##                   'gamma':scipy.stats.expon(scale=10)
#                    'alpha':scipy.stats.expon(scale = 0.0001)
#                   }
#    gs_clf = RandomizedSearchCV(text_clf,parameters,n_jobs=-1,n_iter = 50)
#    gs_clf = gs_clf.fit(X_val_sparse,y_val)
#    gs_clf.best_score_
#    gs_clf.best_params_
