# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 18:10:30 2017

@author: jeesu
"""
from sklearn import svm
import time

from data_input import get_sparse_data

KERNEL='linear'
C_PARAM=15

start_time = time.time() 
[y_train,X_raw_train,X_train_sparse],[y_val,X_raw_val,X_val_sparse],[y_test,X_raw_test,X_test_sparse] = get_sparse_data()

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


"""
Tests for three different multi-class SVC, SVC with linear kernel, LinearSVC, and nuSVC with rbf kernel.
Highest accuracy is SVC with linear kernel with 68.73% on training, 68.89% on validation, 68.09% on Testing.
Accuracies hold around 67% at the moment
"""
linsvc = svm.SVC(C=C_PARAM, kernel=KERNEL, decision_function_shape='ovo')
LinearSVC = svm.LinearSVC(C=C_PARAM)
rbf_svc = svm.NuSVC(decision_function_shape='ovo')

linsvc.fit(X_train_sparse, y_train)
rbf_svc.fit(X_train_sparse, y_train)
LinearSVC.fit(X_train_sparse, y_train)

check_acc(linsvc, "SVC with linear kernel", True)
check_acc(LinearSVC, "LinearSVC", True)
check_acc(rbf_svc, "nuSVC", True)

print(" --- extracting time: %s seconds ---" % (time.time() - start_time))
