# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:47:53 2017

@author: Johnny
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:12:11 2017

@author: Johnny
"""

from sklearn.feature_extraction.text import CountVectorizer
import xlrd
#import bayesian_classifier as bc
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import train_test_split
import numpy as np



TRAIN_SET_LOC = './training_dataset.txt'
TEST_SET_LOC = './test_dataset.xlsx'

def import_data():
#    #load the files
    wb = xlrd.open_workbook(TEST_SET_LOC)
#    wb.sheet_names()
    sh = wb.sheet_by_index(0)
    i = 0
    test_X = []
    test_y = []
    while i < sh.nrows:
        test_X.append(sh.cell(i,0).value)
        test_y.append((sh.cell(i,1).value))
        #write into output.txt 
        i += 1
    wb.release_resources()
        
    with open(TRAIN_SET_LOC,'r', encoding='utf-8') as file:
        lines = file.read().splitlines()
        
    #get the X and ys
    training_data = [i.split('\t', 1) for i in lines]
    
    #split into X and ys
    train_X = []
    train_y = []
    for y,x in training_data:
        train_X.append(x)
        train_y.append(y)
        
#        train_text.append(x)
#        label.append(y)
    
    return train_X,train_y,test_X,test_y

#import data
train_X,train_y,test_X,test_y = import_data()

# use test data as train /test data
#X_train, X_val, y_train, y_val = train_test_split(test_data[0],test_data[1], test_size=0.4,random_state=23)

#build a pipeline
text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')),
                    ('tfidf', TfidfTransformer()),
                    ('clf-svm', SGDClassifier(
                            loss ='hinge',
                            penalty = 'l1',
                            alpha = 1e-5,
                            n_iter = 10,
                            shuffle = True
                            )),])
text_clf_svm.fit(train_X,train_y)




#parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
#              'tfidf__use_idf': (True, False),
#               'clf-svm__alpha': (1e-5,1e-6,1e-7,1e-4)
#               }
#gs_clf = GridSearchCV(text_clf_svm,parameters,n_jobs=2)
#gs_clf = gs_clf.fit(X_train,y_train)
#gs_clf.best_score_
#gs_clf.best_params_


#training acc
result_sk = text_clf_svm.predict(train_X)

true_label =np.array(list(map(int,train_y)))
result_sk = np.array(list(map(int,result_sk)))
acc = np.mean(result_sk == true_label) *100.
print("Train Accuracy %.2f%%" % acc)

##Val acc
#result_sk = text_clf_svm.predict(X_val)
#
#true_label =np.array(list(map(int,y_val)))
#result_sk = np.array(list(map(int,result_sk)))
#acc = np.mean(result_sk == true_label) *100.
#print("Val Accuracy %.2f%%" % acc)


#testing acc
result_sk = text_clf_svm.predict(test_X)


true_label =np.array(list(map(int,test_y)))
result_sk = np.array(list(map(int,result_sk)))
acc = np.mean(result_sk == true_label) *100.
print("Testing Accuracy %.2f%%" % acc)



















