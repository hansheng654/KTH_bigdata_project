# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 20:24:11 2017

@author: Johnny
"""
import tensorflow as tf
import numpy as np
from sklearn.neural_network import BernoulliRBM
from data_input import get_sparse_data
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.utils import to_categorical
from classifiers import check_acc
from keras import regularizers
from keras.layers import LSTM, Conv1D, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

[y_train,X_raw_train,X_train_sparse],[y_val,X_raw_val,X_val_sparse],[y_test,X_raw_test,X_test_sparse] = get_sparse_data(max_df= 0.99,min_df= 0.003)

embedding_vecor_length = 256


#convert into one hot
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)
y_val_hot = to_categorical(y_val)

input_shape = X_train_sparse.get_shape()





model = Sequential()
model.add(Dense(1024,activation='elu',input_dim = input_shape[1]))
model.add(Dropout(0.4))
model.add(Dense(512,activation='elu'))
model.add(Dropout(0.4))
model.add(Dense(256,activation='elu'))
model.add(Dropout(0.4))
model.add(Dense(64,activation='elu'))
model.add(Dense(3, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )


# Train the model, iterating on the data in batches of 32 samples
hist = model.fit(X_train_sparse.toarray(), y_train_hot, epochs=100,
          validation_data=(X_val_sparse.toarray(),y_val_hot),
          batch_size= 1024
          ) 



(loss, accuracy) = model.evaluate(X_test_sparse.toarray(), y_test_hot)
print(accuracy)