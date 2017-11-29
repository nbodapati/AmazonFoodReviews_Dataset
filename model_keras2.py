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

from functools import partial 
import keras.backend as K
from itertools import product

from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import sequence

max_features=None

def create_weights(num_classes=5,distribution=None):
    #distribution is a dictionary of form {int class_num :float weigth_value}
    num_classes=len(distribution.keys())
    weights=np.zeros((num_classes,num_classes))
    if(distribution!=None):
      for class_num,weight in distribution.items():
          weights[class_num,:]=weight    #every other misclassification gets this penalty. 
          weights[class_num,class_num]=0 #zero penalty for correct classification.  

    return weights        

def get_distribution(series=None):
    #series is a pandas series with indices and values.
    if(type(series)==pd.core.series.Series):
        v=series.value_counts()
    elif(type(series)==np.ndarray):
        series=series.flatten()
        series=pd.Series(series,dtype=int)
        v=series.value_counts()
    print(v)
    v=(2-v/v.sum())  #this makes the weights= 1- num_class_samples/total_samples
    #manually set some values
    #v[4]= 1 #since these are the highest in number - if this value is too low, it can 
               #make accuracy low due to little attention paid to this class points. 
    print(v)
    return v.to_dict()


#This function weighs misclassifications according 
#to the class weight associated with it. 
#those that are less in number get a higher priority.
#Scheme of weighting: (1-samples_from_class_c/total_samples)
def w_categorical_crossentropy(y_true, y_pred, weights):
    #y_pred: this has shape: (n_samples,n_classes)
    #y_true: this has shape: (n_samples,n_classes)
    #weights: this has shape: (n_classes,n_classes)
    #Weights have rows = actual/true class and columns representing predicted_class

    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    y_pred_max_mat=K.cast(y_pred_max_mat,K.tf.float32)
    print(y_pred_max_mat.shape,y_true.shape,weights.shape)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])

    return K.categorical_crossentropy(y_pred, y_true) * final_mask


def only_class13(corpus,target):
    df=pd.DataFrame(corpus)
    df['target']=target
    df=df.loc[(df['target']==1) | (df['target']==4)]
    df.loc[df['target']==4,'target']=2
    corpus=(df.iloc[:,:-1]).as_matrix()
    target=np.array(df['target'])
    return corpus,target 

def no_class5(corpus,target):
    df=pd.DataFrame(corpus)
    df['target']=target
    df=df.loc[(df['target']!=5)]
    corpus=(df.iloc[:,:-1]).as_matrix()
    target=np.array(df['target'])
    return corpus,target 


#load the corpus using pickle
def load_data():
    global max_features
    corpus=pickle.load(open("amazon_text_corpus.p", "rb"))
    print("Corpus length: ",len(corpus))
    N=len(corpus)//4 # only using the first quarter.
    corpus=corpus[:N]

    df=pd.read_csv('./Reviews.csv')
    df=df.ix[:N-1,'Score']
    print(df.value_counts())
    target=np.array(df).reshape(-1,1)

    length = len(sorted(corpus,key=len, reverse=True)[0])
    '''
    c=[]
    t=time.time()
    for x in corpus:
        c.append(x+[0]*(length-len(x)))
    print("t= ",time.time()-t)
    corpus=np.array(c)
    '''
    corpus=sequence.pad_sequences(corpus,maxlen=length) 
    #corpus,target=only_class13(corpus,target)
    #corpus,target=no_class5(corpus,target)

    max_features=np.max(corpus) 

    #d=equal_dist_data(dict(data=corpus,labels=target))
    d=shuffle_data(dict(data=corpus,labels=target))
    corpus=d['data']
    target=d['labels']
    target_=target-1

    #convert to one-hot encoding.
    target=keras.utils.to_categorical(target_,num_classes=5)
    data=dict(data=corpus,labels=target)
    print(data.keys()) 
    return data,target_


def equal_dist_data(data_):
    labels=data_['labels']
    data=data_['data']
    
    no_class5=np.array((labels!=5)).reshape(-1,)
    non_class5_data=data[no_class5,:]
    non_class5_target=labels[no_class5]
    data_=dict(data=non_class5_data,labels=non_class5_target)
    return data_

def shuffle_data(data_):
    labels=data_['labels']
    data=data_['data']

    shuffle=np.arange(len(labels))

    data=data[shuffle,:]
    labels=labels[shuffle]
    return dict(data=data,labels=labels)
   
#split data and targets into train,test,val sets.
def train_val_test_split(train_percent=0.6,val_percent=0.01,test_percent=0.6,data_=None):
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

def visualize_model(model):
    plot_model(model,to_file='model.png',show_layer_names=True)

data,target_=load_data()
split,n_train=train_val_test_split(data_=data)

dist=get_distribution(target_[:n_train])
#get weight matrix
weights=create_weights(distribution=dist)

ncce = partial(w_categorical_crossentropy, weights=weights)
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

x_train=split['x_train']
x_val=split['x_val']
x_test=split['x_test']

y_train=split['y_train']
y_val=split['y_val']
y_test=split['y_test']

time_steps=x_train.shape[1]
num_features=1

#'''
x_train=x_train.reshape(x_train.shape[0],-1,1)
x_val=x_val.reshape(x_val.shape[0],-1,1)
x_test=x_test.reshape(x_test.shape[0],-1,1)
#'''
print(x_train.shape,x_val.shape,x_test.shape,y_train.shape,y_val.shape)

model=Sequential()
#model.add(Embedding(max_features, 128))
model.add(LSTM(128,stateful=False,input_shape=(time_steps,num_features))) #hidden memory unit output is a sequence of 128 float values
#model.add(Dropout(0.5))
#model.add(BatchNormalization())
model.add(Dense(5,activation='softmax'))
#model.add(Dense(1,activation='sigmoid'))

rmsprop=keras.optimizers.rmsprop(lr=1e-5)#,decay=0.01)
sgd=keras.optimizers.SGD(lr=0.1)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train,y_train,shuffle=True,batch_size=32,epochs=10,validation_data=(x_val,y_val))
print(model.predict(x_train))
model.save('./Model_Keras2_large.h5')
model.save_weights('./Model_Keras_Weights2_large.h5')



