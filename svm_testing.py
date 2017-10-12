# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 18:10:30 2017

@author: jeesu
"""
from sklearn import svm
import time
import numpy as np
from data_input import get_count_sparse_data

from scipy.stats import uniform as sp_rand
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV

"""
Extensions for jupyter to autoreload code
%load_ext autoreload
%autoreload 2
"""

KERNEL='linear'
C_PARAM=15

this_time = time.time()
start_time = time.time()
[y_train,X_raw_train,X_train_sparse],[y_val,X_raw_val,X_val_sparse],[y_test,X_raw_test,X_test_sparse], vocab = get_count_sparse_data(get_vocab = True)

print(len(y_train + y_val + y_test))

def check_acc(trained_clf,clf_name,val_acc = False):
    #print training and testing acc automatically 
    #training acc
    result_sk = trained_clf.predict(X_train_sparse)
    
    true_label =np.array(list(map(int,y_train)))
    result_sk = np.array(list(map(int,result_sk)))
    acc = np.mean(result_sk == true_label) *100.
    print("  Training Accuracy, %s with tf-idf: %.2f%%" % (clf_name,acc))
    
    #val acc
    if val_acc:
        result_sk = trained_clf.predict(X_val_sparse)
    
        true_label =np.array(list(map(int,y_val)))
        result_sk = np.array(list(map(int,result_sk)))
        acc1 = np.mean(result_sk == true_label) *100.
        print("Validation Accuracy, %s with tf-idf: %.2f%%" % (clf_name,acc1))
    
    #testing acc
    result_sk = trained_clf.predict(X_test_sparse)
    
    
    true_label =np.array(list(map(int,y_test)))
    result_sk = np.array(list(map(int,result_sk)))
    acc2 = np.mean(result_sk == true_label) *100.
    print("   Testing Accuracy, %s with tf-idf: %.2f%%" % (clf_name,acc2))
    print("  Val+Test mean Acc, %s with tf-idf: %.2f%%" % (clf_name,(acc1+acc2)/2))
    """
    precision, recall, _,_ = precision_recall_fscore_support(true_label,result_sk,labels=[0,1,2])
    print("natural precision %.4f, natural recall %.4f" % (precision[0], recall[0]))
    print("pos precision %.4f, pos recall %.4f" % (precision[1], recall[1]))
    print("neg precision %.4f, neg recall %.4f\n" % (precision[2], recall[2]))
    """

"""
Tests for three different SVC
"""
"""
linsvc = svm.SVC(C=C_PARAM, kernel=KERNEL, decision_function_shape='ovr')
linsvc.fit(X_train_sparse, y_train)
check_acc(linsvc, "SVC with linear kernel", True)
print(" --- LinSVC time: %s seconds ---" % (time.time() - this_time))

this_time = time.time()
polySVC = svm.SVC(C=C_PARAM, kernel=KERNEL, degree=3, decision_function_shape='ovr')
polySVC.fit(X_train_sparse, y_train)
check_acc(polySVC, "polySVC", True)
print(" --- PolySVC time: %s seconds ---" % (time.time() - this_time))
"""
"""
potential values:
0.000181729223238
0.000253924273417 (SGDC2)
0.000348012949643 (SGDC1)
"""

this_time = time.time()
sgdc1 = SGDClassifier(loss = 'hinge', alpha = 0.000181729223238, max_iter = 1000, penalty = 'l1')
sgdc1.fit(X_train_sparse, y_train)
check_acc(sgdc1, "hinge loss", True)
print(" --- sgdc time: %s seconds ---" % (time.time() - this_time))

this_time = time.time()
sgdc2 = SGDClassifier(loss = 'log', alpha = 0.000253924273417, max_iter = 1000, penalty = 'l1')
sgdc2.fit(X_train_sparse, y_train)
check_acc(sgdc2, "log loss", True)
print(" --- log time: %s seconds ---" % (time.time() - this_time))

this_time = time.time()
sgdc3 = SGDClassifier(loss = 'modified_huber', alpha = 0.000260727247845, max_iter = 1000, penalty = 'l1')
sgdc3.fit(X_train_sparse, y_train)
check_acc(sgdc3, "modified_huber loss", True)
print(" --- modified_huber time: %s seconds ---" % (time.time() - this_time))
"""
param_grid = {'alpha': sp_rand(loc=0.000348012949643, scale=0.00001)}
rsearch = RandomizedSearchCV(estimator=sgdc1, param_distributions=param_grid, n_iter=500, n_jobs=-1)
rsearch.fit(X_val_sparse, y_val)
print(rsearch)
# summarize the results of the random parameter search
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)
"""
print("\n --- Total time: %s seconds ---" % (time.time() - start_time))
