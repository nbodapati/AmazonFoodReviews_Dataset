from keras.models import Sequential
from keras.layers import Dense,Dropout,Masking,TimeDistributed
from keras.layers import Embedding 
from keras.layers import LSTM
from keras.utils import plot_model
import keras 
from keras.models import load_model

import pickle
import pandas as pd
import numpy as np

import math, random
import os,sys
import time

from functools import partial 
import keras.backend as K
from itertools import product

from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import sequence

from model_bkup import *
from keras_callbacks import *

max_features=None

def process_data(data):
    global max_features
    word_count=data['Word_count']
    summary_vectors=data['Summary_vectors']    
    text_vectors=data['Text_vectors']    
    sum_length =3 #len(sorted(summary_vectors,key=len, reverse=True)[0])
    text_length=500#len(sorted(text_vectors,key=len, reverse=True)[0]) 
    
    n_samples=len(summary_vectors)
    summary_=np.zeros((n_samples,sum_length,1))
    text_=np.zeros((n_samples,text_length,1))
    summary_vectors=sequence.pad_sequences(summary_vectors,maxlen=sum_length) 
    text_vectors=sequence.pad_sequences(text_vectors,maxlen=text_length) 
    max_features=np.max(text_vectors)+1
    print(max_features)
    summary_=summary_vectors.reshape((n_samples,sum_length))
    text_=text_vectors.reshape((n_samples,text_length))
    return dict(text=text_,summary=summary_) 

def undersample_class(corpus,targets,class_num=4):
    corpus=corpus.reshape(corpus.shape[0],-1)

    df=pd.DataFrame(corpus)
    df['targets']=targets
    
    only5=df[df['targets']==4]
    only5_len=only5.shape[0]
    without5=df[df['targets']!=4]
    only5=only5[:only5_len//4]
    corpus=without5#pd.concat([only5,without5]).sample(frac=1)
    print(corpus.head())
    corpus=corpus.as_matrix()
    
    return (corpus[:,:-1],corpus[:,-1])

def shuffle_data(data_):
    labels=data_['labels']
    data=data_['data']

    shuffle=np.arange(len(labels))

    data=data[shuffle,:]
    labels=labels[shuffle]
    return dict(data=data,labels=labels)

def load_data():
    data=pickle.load(open("dataset.p", "rb"))
    corpus=process_data(data)

    target=np.array(data['Scores']).reshape(-1,1)
    #corpus=corpus['summary']
    corpus=corpus['text']
    target_=target-1

    data_=shuffle_data(data_=dict(data=corpus,labels=target))
    corpus=data_['data']
    target=data_['labels'] 
    print(corpus.shape) 

    n_samples=(corpus.shape[0])
    n_steps=corpus.shape[1] 
    #corpus,target_=undersample_class(corpus,target_)
    #corpus=corpus.reshape(-1,n_steps,1)
    #dist=get_distribution(target_)
    #weights=create_weights(distribution=dist)
    weights=0 
    #convert to one-hot encoding.
    target=keras.utils.to_categorical(target_,num_classes=5)
    data=dict(data=corpus,labels=target)
    print(data.keys()) 
    return data,weights

#split data and targets into train,test,val sets.
def train_val_test_split(train_percent=0.2,val_percent=0.01,test_percent=0.6,data_=None):
    labels=data_['labels']
    data=data_['data']

    n_train=int(train_percent*data.shape[0])
    n_val=int(val_percent*data.shape[0])
    n_test=int(test_percent*data.shape[0])

    x_train=data[0:n_train,:]
    y_train=labels[0:n_train,:]

    x_val=data[n_train:n_train+n_val,:]
    y_val=labels[n_train:n_train+n_val,:]

    x_test=data[n_train+n_val:,:]
    y_test=labels[n_train+n_val:,:]

    split=dict(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,x_val=x_val,y_val=y_val)
    return split,n_train


data,weights=load_data()
split,n_train=train_val_test_split(data_=data)
#callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
ncce = partial(w_categorical_crossentropy, weights=weights)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.001)

prob_logger=keras.callbacks.ProgbarLogger(count_mode='samples')


x_train=split['x_train']
x_val=split['x_val']
x_test=split['x_test']

y_train=split['y_train']
y_val=split['y_val']
y_test=split['y_test']

time_steps=x_train.shape[1]
num_features=1
histories=Histories((x_val,y_val))

print(x_train.shape,x_val.shape,x_test.shape,y_train.shape,y_val.shape)

model=Sequential()
model.add(Embedding(max_features,128))
model.add(Masking(mask_value=0.0,input_shape=(time_steps,num_features)))

model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2)) #hidden memory unit output is a sequence of 128 float values
model.add(Dense(5,activation='softmax'))

rmsprop=keras.optimizers.rmsprop(lr=1e-5)#,decay=0.01)
sgd=keras.optimizers.adam(lr=10,decay=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model=load_model('./Model_Keras_large.h5')
model.load_weights('./Model_Keras_Weights_large.h5')

model.fit(x_train,y_train,shuffle=False,\
          batch_size=32,epochs=10,\
          validation_data=(x_val,y_val),\
          callbacks=[reduce_lr,prob_logger] )
model.save('./Model_Keras_large.h5')
model.save_weights('./Model_Keras_Weights_large.h5')


