import pandas as pd
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, Dropout, MaxPooling1D, Convolution1D, Conv1D
from keras.layers import GRU, merge, Lambda
from keras.layers import Embedding, TimeDistributed, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
import numpy as np
import tensorflow as tf
import re
from keras import backend as K
import keras.callbacks
import sys
import os
import pyexcel as pyexcel
import json

from sklearn.cross_validation import train_test_split

def binarize(x, sz=0):
    sz = len_chars
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))

def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], len_chars




data = pd.read_csv('data.csv')

txt = ''
docs = [] #body, line by line
labels = [] #label

for content, label in zip(data.Headline, data.Label):
    docs.append(content)  
    labels.append(label)

num_sent = []
for doc in docs:
    num_sent.append(len(doc))   
    txt += doc

maxlen = len(max(docs, key = len)) #max length of an element in docs list
print("maxlen: ", maxlen)
chars = set(txt)
global len_chars
len_chars = len(chars)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
##
##
##
X = np.ones((len(docs), maxlen), dtype=np.int64) * -1
Y = np.array(labels)
##    
for i, doc in enumerate(docs):
   for t, char in enumerate(doc):
       X[i, t] = char_indices[char]   
##
##
print ("X-shape: ", X.shape)
print ("Y-shape: ", Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
#
## shuffle
ids = np.arange(len(X_train))
np.random.shuffle(ids)
X_train = X_train[ids]
Y_train = Y_train[ids]

print ("X-train shape: ", X_train.shape)
print ("Y-train shape: ", Y_train.shape)
###

#Char embedding using CNN start
filter_length = [5, 3, 3]
nb_filter = [196, 196, 300]
pool_size = 2
#
in_sentence = Input(shape=(maxlen,), dtype='int64')
#
embedded = Lambda(binarize, output_shape=binarize_outshape)(in_sentence)
print("embedded after lambda:",embedded)
#
for i in range(len(nb_filter)):
    embedded = Conv1D(activation="relu", filters=nb_filter[i], kernel_size=3, strides=1, padding="valid", kernel_initializer="glorot_normal")(embedded)
    embedded = Dropout(0.1)(embedded)
    embedded = MaxPooling1D(pool_size=pool_size)(embedded)

####Char embedding using CNN end

forward_sent = GRU(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, implementation=2)(embedded)
backward_sent = GRU(128, return_sequences=False, go_backwards=True, dropout=0.2, recurrent_dropout=0.2, implementation=2)(embedded)

sent_encode = Concatenate()([forward_sent, backward_sent])
sent_encode = Dropout(0.3)(sent_encode)
output = Dense(1, activation='sigmoid')(sent_encode)

model = Model(input=in_sentence, output=output)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')

batch_size = 16
model.fit(X_train, Y_train, batch_size=batch_size, epochs=10,
          validation_split=0.1, callbacks=[earlystop_cb])
score, acc = model.evaluate(X_test, Y_test,
                            batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)

def predict_classes(model, X_test):
    proba = model.predict(X_test)
    if proba.shape[-1] > 1:
        return proba.argmax(axis=-1)
    else:
        return (proba > 0.5).astype('int32')

y_pred = predict_classes(model, X_test)
y_scores = model.predict(X_test)

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
roc = roc_auc_score(Y_test, y_scores)
print('ROC score:', roc)

metrics = classification_report(Y_test, y_pred, digits=4)
print('Classification Report \n')
print(metrics)

cm = confusion_matrix(Y_test, y_pred)
print('Confusion Matrix \n')
print(cm)

print ("********************************FINISH**************************************")








