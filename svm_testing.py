# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 18:10:30 2017

@author: jeesu
"""
from sklearn import svm
import time

from data_input import get_sparse_data
import classifiers

KERNEL='linear'
C_PARAM=15

[y_train,X_raw_train,X_train_sparse],[y_val,X_raw_val,X_val_sparse],[y_test,X_raw_test,X_test_sparse] = get_sparse_data()

linsvc = svm.SVC(C=C_PARAM, kernel='linear', decision_function_shape='ovo')
LinearSVC = svm.LinearSVC(C=C_PARAM)
rbf_svc = svm.NuSVC(decision_function_shape='ovo')

linsvc.fit(X_train_sparse, y_train)
rbf_svc.fit(X_train_sparse, y_train)
LinearSVC.fit(X_train_sparse, y_train)

classifiers.check_acc(linsvc, "SVC with linear kernel", True)
classifiers.check_acc(LinearSVC, "LinearSVC", True)
classifiers.check_acc(rbf_svc, "nuSVC", True)

start_time = time.time() 
print(" --- extracting time: %s seconds ---" % (time.time() - start_time))