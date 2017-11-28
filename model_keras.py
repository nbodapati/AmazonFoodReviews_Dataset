from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import Embedding 
from keras.layers import LSTM
from keras.utils import plot_model
import keras 

import pickle
import pandas as pd
import numpy as np

import math, random
import os,sys
import time


#load the corpus using pickle
def load_data():
    corpus=pickle.load(open( "amazon_text_corpus.p", "rb" ))
    print("Corpus length: ",len(corpus))
    N=len(corpus)//4 # only using the first quarter.
    corpus=corpus[:N]

    length = len(sorted(corpus,key=len, reverse=True)[0])

    c=[]
    t=time.time()
    for x in corpus:
        c.append(x+[0]*(length-len(x)))
    print("t= ",time.time()-t)
    corpus=np.array(c)
    
    df=pd.read_csv('./Reviews.csv')
    df=df['Score']
    target=np.array(df).reshape(-1,1)
    target=target-1
    #convert to one-hot encoding.
    target=keras.utils.to_categorical(target,num_classes=5)
    data=dict(data=corpus,labels=target)
    print(data.keys()) 
    return data

#split data and targets into train,test,val sets.
def train_val_test_split(train_percent=0.01,val_percent=0.02,test_percent=0.6,data_=None):
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
    return split

def visualize_model(model):
    plot_model(model,to_file='model.png',show_layer_names=True)

data=load_data()
split=train_val_test_split(data_=data)

x_train=split['x_train']
x_val=split['x_val']
x_test=split['x_test']

y_train=split['y_train']
y_val=split['y_val']
y_test=split['y_test']

time_steps=x_train.shape[1]
num_features=1

x_train=x_train.reshape(x_train.shape[0],-1,1)
x_val=x_val.reshape(x_val.shape[0],-1,1)
x_test=x_test.reshape(x_test.shape[0],-1,1)
print(x_train.shape,x_val.shape,x_test.shape)

model=Sequential()
model.add(LSTM(128,stateful=False,input_shape=(time_steps,num_features))) #hidden memory unit output is a sequence of 128 float values
model.add(Dropout(0.5))
model.add(Dense(5,activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=32,epochs=10,validation_data=(x_val,y_val))
score=model.evaluate(x_test,y_test,batch_size=16)

visualize_model(model)



