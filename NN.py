# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 20:24:11 2017

@author: Johnny
"""
import tensorflow as tf
import numpy as np
from data_input import get_sparse_data
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.utils import to_categorical
from classifiers import check_acc
from keras import regularizers

[y_train,X_raw_train,X_train_sparse],[y_val,X_raw_val,X_val_sparse],[y_test,X_raw_test,X_test_sparse] = get_sparse_data(max_df= 0.999,min_df= 0.0001)

#convert -1 into 2
y_train = [2 if x==-1 else x for x in y_train]
y_test = [2 if x==-1 else x for x in y_test]
y_val = [2 if x==-1 else x for x in y_val]
#convert into one hot
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)
y_val_hot = to_categorical(y_val)

input_shape = X_train_sparse.get_shape()


reg = regularizers.l1(1)

model = Sequential()
model.add(Dense(1024, activation='relu', input_dim=input_shape[1],
                activity_regularizer= reg,
                kernel_regularizer=reg))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu', input_dim=1024,
                activity_regularizer= reg,
                kernel_regularizer=reg))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', input_dim=1024,
                activity_regularizer= reg,
                kernel_regularizer=reg))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', input_dim=512,
                activity_regularizer= reg,
                kernel_regularizer=reg))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu', input_dim=128,
                activity_regularizer= reg,
                kernel_regularizer=reg))
model.add(Dropout(0.5))
model.add(Dense(3, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )


# Train the model, iterating on the data in batches of 32 samples
hist = model.fit(X_train_sparse.toarray(), y_train_hot, epochs=9,
          validation_data=(X_val_sparse.toarray(),y_val_hot),
          batch_size= 1024
          ) 

#loss_and_metrics = model.evaluate(X_test_sparse.toarray(), y_test_hot)


classes = model.predict(X_test_sparse.toarray())
result = np.argmax(classes,axis=1)
acc = np.mean(result == y_test) *100.
print(acc)