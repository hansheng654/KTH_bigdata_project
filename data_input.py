# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 22:28:20 2017

@author: Johnny
"""

import glob
import os
import re
import string
#from nltk.corpus import stopwords
from porter2stemmer import Porter2Stemmer
from nltk import bigrams
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer 
from sklearn.utils import shuffle
import numpy as np
from sys import platform
from keras.utils import to_categorical
import pickle
import pandas as pd 


VADER_RAW_PATH ='\data\VADER\*_GroundTruth.txt'
#Million_TWEETS = '\\data\\for_final_project_V.txt' #to be used
Millionaire = 'cleaned_million'

Mobile_tweets = '100kPhoneTweets.csv' #to be added
use_biagrams = False  #whether to use biagrams as features


def _import_data():
    """ Function used by get_data methods
    Returns X and Y as raw text.
    Y value are generated by the sentiment_converter, which maps -4 to 4 sentiment into 0,1,2
    
    
    """
    cwd = os.getcwd()
    Y = []
    X = []
    if platform=='win32':
        data_path = cwd+"\\"+VADER_RAW_PATH
    else:
        data_path = cwd+'/'+ VADER_RAW_PATH.replace('\\','/')
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
        if y > pos_thresh_hold: #happy
            return 1
        elif y < neg_thresh_hold: #sad
            return 0
        else:
            return -1 
        #snap into 0,1,2 based on thresh holds
    Y = list(map(sentiment_converter,Y))
    my_X = []
    my_Y = []  
    lent = len(Y)

    for i in range(lent):
        if Y[i] == -1 :
            continue
        else:
            my_X.append(X[i])
            my_Y.append(Y[i])
            

    
    return my_X,my_Y

def become_millionaire():
    """ Call this function to become a millionaire - someone has 1 million dataset to play with
    
    Return:
        cleaned 1 million data in a list
    
    """
#    count = 0
#    cwd = os.getcwd()
#    X = []
#    if platform=='win32':
#        data_path =X cwd+Million_TWEETS
#    else:
#        data_path = cwd+'/'+ Million_TWEETS.replace('\\','/')
#    with open(data_path, 'rb') as f:
#            for line in f.readlines():
#                X.append(_input_cleaning(line.decode()))
#                if(count % 1000 ==0):
#                    print("progress:",count/10000.0)    
#                count+=1
#                
#    del X[0]
    with open (Millionaire, 'rb') as fp:
        X = pickle.load(fp)
    
    return X

def phone_dataset():
    X = pd.read_csv(Mobile_tweets ,encoding = "cp1251",delimiter=',',header = None)
    X = list(X[0])
    iphone_tweets= []
    samsung_tweets =  []
    
    for text in X:
        cleaned_text = _input_cleaning(str(text))
        if any(word in cleaned_text for word in iphone):
            iphone_tweets.append(cleaned_text)
        if any(word in cleaned_text for word in samsung):
            samsung_tweets.append(cleaned_text)
    return samsung_tweets,iphone_tweets
        
        
        
    
    


# for removing punctuation
puncList = [".", "!", "?", ",", ";", ":", "-", "'", "\"", 
                "!!", "!!!", "??", "???", "?!?", "!?!", "?!?!", "!?!?",'..','...'] 
regex_remove_punctuation = re.compile('[%s]' % re.escape(string.punctuation + ''.join(puncList)))
def _input_cleaning(text):
    """ a function used by Vectorizer
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

#    for text in raw_X:
    text = text.lower()
    text_mod = regex_remove_punctuation.sub('', text) # removes punctuation (but loses emoticons & contractions)
    wordsOnly = str(text_mod).split()

    # get rid of residual empty items or single letter "words" like 'a' and 'I' from wordsAndEmoticons
    stemmer = Porter2Stemmer()
    stemed_cleaned = []
    for word in wordsOnly:
        if len(word) <= 1:
            continue
#            wordsOnly.remove(word)    
        #remove httpsimport Stemmer
        elif word.find('http') > -1:
#            wordsOnly.remove(word)
            continue
#        elif word in stopwords.words('english'): #Too slow for our dataset
#            continue
        else:
            stemed_cleaned.append(stemmer.stem(word))
        
    if use_bigrams:
    #02/10 - bigram!
        def bigram_word_feats(words):
            if len(words) < 1:
                return []
            bigram = bigrams(words)
            return [' '.join(ngram) for ngram in bigram]
            #return bigrams#dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])
        biagram_pairs = bigram_word_feats(stemed_cleaned)  
        return ' '.join(biagram_pairs)
    else:
        return ' '.join(stemed_cleaned)

def get_data(train_split = 0.6, val_split = 0.3,one_hot = True):
    '''Get the training, validation and testing data based on a split
    The returned data is NOT vectorised 
    
    The size of testing set is the remaining portion of the total dataset
    can be 0
    
    - auto shuffle
    
    Parameters:
        train_split: float, the percentage of training set, default to 0.6
        val_split: float, the percentage of validation set, default to 0.3
        one_hot: bool, indicate whether to use one_hot encoding on y

    Returns:
        [y_train,X_raw_train,X_train_clean],
        [y_val,X_raw_val,X_val_clean],
        [y_test,X_raw_test,X_test_clean]
    
    Raises:
        Error if train + val split > 1.0
    '''
    assert train_split + val_split <= 1.0
    
    raw_X,raw_Y = _import_data()
    cleaned_X = []
    for text in raw_X:
        cleaned_X.append(_input_cleaning(text))
    
    raw_X, cleaned_X, y = shuffle(raw_X, cleaned_X,raw_Y)
    
    
    m = np.size(y)
    if one_hot:
        y = to_categorical(y)
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
    

def get_sparse_data(train_split = 0.6, val_split = 0.3,max_df = 0.995, min_df = 0.001,one_hot = False):
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
        one_hot: bool, indicate whetehr to use one-hot encoding, default to False
    
    Returns:
        [y_train,X_raw_train,X_train_sparse],[y_val,X_raw_val,X_val_sparse],
        [y_test,X_raw_test,X_val_test]
    
    Raises:
        Error if train + val split >1.0
    
    """
    assert train_split + val_split <= 1.0

    #y ranges from -4 to 4            
    raw_X,raw_Y = _import_data()
    #tfidf vectorizer
    transformer = TfidfVectorizer(preprocessor=_input_cleaning,lowercase = False,max_df = max_df,
                                  min_df = min_df)
    X_sparse = transformer.fit_transform(raw_X)
    raw_X, X_sparse, y = shuffle(raw_X, X_sparse,raw_Y)
    
    m = np.size(y)
    if one_hot:
        y = to_categorical(y)
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

def get_count_sparse_data(train_split = 0.6, val_split = 0.3,max_df = 0.995, min_df = 0.001,one_hot = False,get_vocab  = False):
    """ Get the training, validation and testing data based on a split
    The size of testing set is the remaining portion of the total dataset
    can be 0
    
    - uses CountVectorizer, X values are the word count of a feature word dictionary
    - auto shuffle
    - max and min df controls the dimension of X
    - use 2 as ngram_range
    
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
    raw_X,raw_Y = _import_data()
    
    """
    #uncomment to transform the three classes into two, positive and negative
    for index, x in enumerate(raw_X):
        if raw_Y[index] == 0:
            del raw_X[index]
            del raw_Y[index]
    """
    #tfidf vectorizer
    transformer = CountVectorizer(preprocessor=_input_cleaning,lowercase = False,max_df = max_df,
                                  min_df = min_df,ngram_range = (1,2))
    X_sparse = transformer.fit_transform(raw_X)
    raw_X, X_sparse, y = shuffle(raw_X, X_sparse,raw_Y)
    
    m = np.size(y)
    if one_hot:
        y = to_categorical(y)
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
    
    if get_vocab:
        return [y_train,X_raw_train,X_train_sparse],[y_val,X_raw_val,X_val_sparse],[y_test,X_raw_test,X_test_sparse],transformer.vocabulary_
    else:
        return [y_train,X_raw_train,X_train_sparse],[y_val,X_raw_val,X_val_sparse],[y_test,X_raw_test,X_test_sparse]


     
def batch_iter(data, batch_size, num_epochs, shuffle=True): 
    ''' taken from http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
    used by tensorflow
    '''
    #Generates a batch iterator for a dataset.    
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


    

if __name__ == '__main__':
#    [y_train,X_raw_train,X_train_sparse],[y_val,X_raw_val,X_val_sparse],[y_test,X_raw_test,X_test_sparse] = get_sparse_data()
     [y_train,X_raw_train,X_train_clean],[y_val,X_raw_val,X_val_clean],[y_test,X_raw_test,X_test_clean] = get_data()
#     [y_train,X_raw_train,X_train_clean],[y_val,X_raw_val,X_val_clean],[y_test,X_raw_test,X_test_clean] = get_count_sparse_data()
#    X = become_millionaire()
#     = get_data()