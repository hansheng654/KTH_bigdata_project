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

start_time = time.time() 
[y_train,X_raw_train,X_train_sparse],[y_val,X_raw_val,X_val_sparse],[y_test,X_raw_test,X_test_sparse] = get_sparse_data()

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

classifiers.check_acc(linsvc, "SVC with linear kernel", True)
classifiers.check_acc(LinearSVC, "LinearSVC", True)
classifiers.check_acc(rbf_svc, "nuSVC", True)

print(" --- extracting time: %s seconds ---" % (time.time() - start_time))
