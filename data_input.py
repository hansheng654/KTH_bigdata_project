# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 22:28:20 2017

@author: Johnny
"""

#import csv
import glob
import os
import re
import string
from nltk.stem.lancaster import LancasterStemmer
from sklearn.model_selection import train_test_split
#from nltk.corpus import stopwords
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
import numpy as np

VADER_RAW_PATH ='\data\VADER\*_GroundTruth.txt'
Million_TWEETS = '\data\for_final_project_V.txt'


def import_data():
    cwd = os.getcwd()
    Y = []
    X = []
    data_path = cwd+"\\"+VADER_RAW_PATH
    for filename in glob.glob(data_path):
        with open(filename, 'r',encoding='utf-8') as f:
            for line in f.readlines():
                col = line.split('\t')
                Y.append(float(col[1]))
                X.append(col[2])
    
    def sentiment_converter(y):
        neg_thresh_hold = -1
        pos_thresh_hold = 0.6
        y = float(y)
        if y > pos_thresh_hold:
            return 1
        elif y < neg_thresh_hold:
            return -1
        return 0 
        #snap into -1,0,or 1 based on thresh holds
    Y = list(map(sentiment_converter,Y))
    
    return X,Y



def _input_cleaning(text):
    """ a function used by TfidfVectorizer
    from raw input texts, remove stop words, perform steming, 
    remove punctuation, urls and emojis
    
    some code are from VADER:
      Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for 
      Sentiment Analysis of Social Media Text. Eighth International Conference on 
      Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
    
    parameters:
        text: string, a input string
    
    returns:
        cleaned text, removed punctuations, added stemming, removed urls 
    
    """
    # for removing punctuation
    regex_remove_punctuation = re.compile('[%s]' % re.escape(string.punctuation))
    puncList = [".", "!", "?", ",", ";", ":", "-", "'", "\"", 
                    "!!", "!!!", "??", "???", "?!?", "!?!", "?!?!", "!?!?"] 
#    for text in raw_X:
    text = text.lower()
    wordsAndEmoticons = str(text).split() #doesn't separate words from adjacent punctuation (keeps emoticons & contractions)
    text_mod = regex_remove_punctuation.sub('', text) # removes punctuation (but loses emoticons & contractions)
    wordsOnly = str(text_mod).split()
    # get rid of empty items or single letter "words" like 'a' and 'I' from wordsOnly
    for word in wordsOnly:
        if len(word) <= 1:
            wordsOnly.remove(word)    
    # now remove adjacent & redundant punctuation from [wordsAndEmoticons] while keeping emoticons and contractions
    for word in wordsOnly:
        for p in puncList:
            pword = p + word
            x1 = wordsAndEmoticons.count(pword)
            while x1 > 0:
                i = wordsAndEmoticons.index(pword)
                wordsAndEmoticons.remove(pword)
                wordsAndEmoticons.insert(i, word)
                x1 = wordsAndEmoticons.count(pword)
            
            wordp = word + p
            x2 = wordsAndEmoticons.count(wordp)
            while x2 > 0:
                i = wordsAndEmoticons.index(wordp)
                wordsAndEmoticons.remove(wordp)
                wordsAndEmoticons.insert(i, word)
                x2 = wordsAndEmoticons.count(wordp)
    # get rid of residual empty items or single letter "words" like 'a' and 'I' from wordsAndEmoticons
    stemmer = LancasterStemmer()
    stemed_cleaned = []
    for word in wordsAndEmoticons:
        if len(word) <= 1:
            wordsAndEmoticons.remove(word)    
        #remove https
        elif word.find('http') > -1:
            wordsAndEmoticons.remove(word)
        else:
            stemed_cleaned.append(stemmer.stem(word))
        #not removing it,because some words are important!
#            #remove stopwords 
#            elif word in stopwords.words('english'):
#                wordsAndEmoticons.remove(word)
    return ' '.join(stemed_cleaned)

def get_data(train_split = 0.6, val_split = 0.3):
    '''Get the training, validation and testing data based on a split
    The returned data is NOT vectorised 
    
    The size of testing set is the remaining portion of the total dataset
    can be 0
    
    - auto shuffle
    
    Parameters:
        train_split: float, the percentage of training set, default to 0.6
        val_split: float, the percentage of validation set, default to 0.3

    Returns:
        [y_train,X_raw_train,X_train_clean],
        [y_val,X_raw_val,X_val_clean],
        [y_test,X_raw_test,X_test_clean]
    
    Raises:
        Error if train + val split > 1.0
    '''
    assert train_split + val_split <= 1.0
    
    raw_X,raw_Y = import_data()
    cleaned_X = []
    for text in raw_X:
        cleaned_X.append(_input_cleaning(text))
    
    raw_X, cleaned_X, y = shuffle(raw_X, cleaned_X,raw_Y)
    
    m = np.size(y)
    train_p = int(np.floor(train_split * m))
    val_p = int(np.floor((val_split * m) + train_p))
    
    y_train = y[0:train_p]
    X_raw_train = raw_X[0:train_p]
    X_train_clean = cleaned_X[0:train_p]
    
    y_val = y[train_p:val_p]
    X_raw_val = raw_X[train_p:val_p]
    X_val_clean= cleaned_X[train_p:val_p]
    
    y_test = y[val_p:m]
    X_raw_test = raw_X[val_p:m]
    X_test_clean = cleaned_X[val_p:m]
    
    return [y_train,X_raw_train,X_train_clean],[y_val,X_raw_val,X_val_clean],[y_test,X_raw_test,X_test_clean]
    

def get_sparse_data(train_split = 0.6, val_split = 0.3,max_df = 0.995, min_df = 0.001):
    """ Get the training, validation and testing data based on a split
    
    The size of testing set is the remaining portion of the total dataset
    can be 0
    
    - Uses TF-IDF Vectoriser
    - auto shuffle
    - max and min df controls the dimension of X
    
    Parameters:
        train_split: float, the percentage of training set, default to 0.6
        val_split: float, the percentage of validation set, default to 0.3
        max_df: float, the upper boundry for word frequencies, default to 0.995
        min_df: float, the lower boundry for word frequencies, default to 0.001
    
    Returns:
        [y_train,X_raw_train,X_train_sparse],[y_val,X_raw_val,X_val_sparse],
        [y_test,X_raw_test,X_val_test]
    
    Raises:
        Error if train + val split >1.0
    
    """
    assert train_split + val_split <= 1.0

    #y ranges from -4 to 4            
    raw_X,raw_Y = import_data()
    #tfidf vectorizer
    transformer = TfidfVectorizer(preprocessor=_input_cleaning,lowercase = False,max_df = max_df,
                                  min_df = min_df)
    X_sparse = transformer.fit_transform(raw_X)
    raw_X, X_sparse, y = shuffle(raw_X, X_sparse,raw_Y)
    
    m = np.size(y)
    train_p = int(np.floor(train_split * m))
    val_p = int(np.floor((val_split * m) + train_p))
    
    y_train = y[0:train_p]
    X_raw_train = raw_X[0:train_p]
    X_train_sparse = X_sparse[0:train_p,:]
    
    y_val = y[train_p:val_p]
    X_raw_val = raw_X[train_p:val_p]
    X_val_sparse = X_sparse[train_p:val_p,:]
    
    y_test = y[val_p:m]
    X_raw_test = raw_X[val_p:m]
    X_test_sparse = X_sparse[val_p:m,:]
    
    return [y_train,X_raw_train,X_train_sparse],[y_val,X_raw_val,X_val_sparse],[y_test,X_raw_test,X_test_sparse]
    
if __name__ == '__main__':
#    [y_train,X_raw_train,X_train_sparse],[y_val,X_raw_val,X_val_sparse],[y_test,X_raw_test,X_test_sparse] = get_sparse_data()
     [y_train,X_raw_train,X_train_clean],[y_val,X_raw_val,X_val_clean],[y_test,X_raw_test,X_test_clean] = get_data()