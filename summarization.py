from keras.models import Sequential
from keras.layers import Dense,Dropout,Masking,TimeDistributed
from keras.layers import Embedding 
from keras.layers import LSTM,Activation,RepeatVector

from keras.regularizers import l2
import keras 
from keras.models import load_model

import pickle
import pandas as pd
import numpy as np

import math, random
import os,sys
import time

from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import sequence

from model_bkup import *
from keras_callbacks import *
from keras.models import Model
from keras.models import model_from_json
from cleaning_summary import *

max_features=None
vocab_size=None

    
def process_data(data):
    summary_count=data['Summary_count']
    summary_rank=data['Summary_rank']
    summary_vectors=tokens_to_vectors(data['Summary_tokens'],\
        summary_count,summary_rank)
    #summary_vectors=summary_vectors.reshape(len(summary_vectors),-1)
 
    text_count=data['Text_count']
    text_rank=data['Text_rank']
    text_vectors=tokens_to_vectors(data['Text_tokens'],\
        text_count,text_rank)
    #text_vectors=text_vectors.reshape(len(text_vectors),-1)
    
    print(summary_vectors[:5])
    print(text_vectors[:5])
    sum_length =15 
    text_length=500 
    
    n_samples=len(summary_vectors)
    summary_vectors=sequence.pad_sequences(summary_vectors,maxlen=sum_length) 
    text_vectors=sequence.pad_sequences(text_vectors,maxlen=text_length) 
   
    #vocab size for the time-distributed/softmax layer.
    summary_=summary_vectors#.reshape((n_samples,sum_length))
    text_=text_vectors#.reshape((n_samples,text_length))
    return dict(text=text_,summary=summary_,text_tokens=data['Text_tokens'],summary_tokens=data['Summary_tokens']) 

def shuffle_data(data_):
    head=data_['head']
    desc=data_['desc']
    target=data_['t']
   
    shuffle=np.arange(len(head))

    head=head[shuffle,:]
    desc=desc[shuffle,:]
    target=target[shuffle]
    return dict(head=head,desc=desc,t=target,\
                text_tokens=data_['text_tokens'],\
                       summary_tokens=data_['summary_tokens'])

def load_data():
    global max_features,vocab_size
    data=pickle.load(open("dataset2.p", "rb"))

    #keys: Scores,Summary_vectors,Text_vectors
    #Text_count,Summary_count
    corpus=process_data(data)
    target=np.array(data['Scores']).reshape(-1,1)
    head=corpus['summary']
    desc=corpus['text']

    max_features=np.max(desc)+1
    vocab_size=np.max(head)+1#len(data['Summary_count'].keys())+1 
    print("Max features: ",max_features,"vocab_size: ",vocab_size)

    print("HEad shape: ",head.shape)
    data_=shuffle_data(data_=dict(head=head,desc=desc,t=target,text_tokens=corpus['text_tokens'],\
                       summary_tokens=corpus['summary_tokens']))
    return data_

#split data and targets into train,test,val sets.
def train_val_test_split(train_percent=0.1,val_percent=0.08,test_percent=0.6,data_=None):
    global vocab_size
    print(data_.keys()) 
    text_tokens=data_['text_tokens']
    summary_tokens=data_['summary_tokens'] 
    head=data_['head']
    desc=data_['desc']
    target=data_['t']

    n_train=int(train_percent*head.shape[0])
    n_val=int(val_percent*head.shape[0])
    n_test=int(test_percent*head.shape[0])

    x_train=desc[0:n_train,:]
    y_train=head[0:n_train,:]
    y_train=keras.utils.to_categorical(y_train,num_classes=vocab_size+1)
    z_train=target[0:n_train]
    text_train=text_tokens[:n_train]
    summary_train=summary_tokens[:n_train]

    x_val=desc[n_train:n_train+n_val,:]
    y_val=head[n_train:n_train+n_val,:]
    y_val=keras.utils.to_categorical(y_val,num_classes=vocab_size+1)
    z_val=target[n_train:n_train+n_val]
    text_val=text_tokens[n_train:n_train+n_val]
    summary_val=summary_tokens[n_train:n_train+n_val]

    x_test=desc[n_train+n_val:,:]
    y_test=head[n_train+n_val:,:]
    #y_test=keras.utils.to_categorical(y_test,num_classes=vocab_size+1)
    z_test=target[n_train+n_val:]
    text_test=text_tokens[n_train+n_val:]
    summary_test=summary_tokens[n_train+n_val:]

    split=dict(x_train=x_train,y_train=y_train,z_train=z_train,text_train=text_train,summary_train=summary_train,x_test=x_test,y_test=y_test,z_test=z_test,\
                      text_test=text_test,summary_test=summary_test,\
                      x_val=x_val,y_val=y_val,z_val=z_val,text_val=text_val,summary_val=summary_val)
    return split

def predict(x_train,y_train,z_train,\
            x_val,y_val,z_val,\
            x_test,y_test,z_test,model,layer_num=3):

    from keras.models import model_from_json
    import json
    json_file = open('model_summary.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("./Model_Keras_Weights_summary.h5")
    print("Loaded model from disk")

    lstm_layer_model = Model(inputs=loaded_model.input,
                                 outputs=loaded_model.get_layer(index=layer_num).output)
    lstm_output = lstm_layer_model.predict(x_val)
    print("Lstm output: ",lstm_output,lstm_output.shape)   

    lstm_layer_model = Model(inputs=loaded_model.input,
                                 outputs=loaded_model.get_layer(index=layer_num+1).output)
    lstm_output2 = lstm_layer_model.predict(x_val)
    y_pred=np.argmax(lstm_output2,axis=1)
    y_val1=np.argmax(y_val,axis=1)

    pickle.dump(dict(lstm_output=lstm_output.tolist(),y_pred=list(y_pred),y_val=list(y_val1)),\
                    open('output_summary.pkl','wb'))
         
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = loaded_model.evaluate(x_val, y_val, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

data=load_data()
split=train_val_test_split(data_=data)

#print("TRaining on..",split['text_train'])
#print(split['summary_train'])
#callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.001)

prob_logger=keras.callbacks.ProgbarLogger(count_mode='samples')

x_train=split['x_train']
x_val=split['x_val']
x_test=split['x_test']

y_train=split['y_train']

y_val=split['y_val']


z_train=split['z_train']
z_val=split['z_val']
z_test=split['z_test']

time_steps=x_train.shape[1] #500
num_features=128
histories=Histories((x_val,y_val),'summary_losses.pkl','model_summary.json',\
                   './Model_Keras_Weights_summary.h5')

print(x_train.shape,x_val.shape,x_test.shape,y_train.shape,y_val.shape)

weight_decay=0
reg=l2(weight_decay) if weight_decay else None

def neural_net_model1():
    global max_features,vocab_size
    #Architecture:
    #LSTM->LSTM->Softmax 
    #Not encoder-decoder architecture.
    model=Sequential()
    model.add(Embedding(max_features,128,name='embedding'))
    model.add(Masking(mask_value=0.0,input_shape=(time_steps,num_features),name='masking'))
    model.add(LSTM(128, dropout=0.2,return_sequences=False,\
                    recurrent_dropout=0.2,name='lstm_1'))

    model.add(RepeatVector(15)) 
    model.add(LSTM(128, dropout=0.2,return_sequences=True,\
                    recurrent_dropout=0.2,name='lstm_2'))
    model.add(TimeDistributed(Dense(vocab_size+1,name = 'timedistributed_1')))
    model.add(Activation('softmax', name='activation_1'))
    print(model.summary())
    return model

model=neural_net_model1()
'''
json_file = open('model_summary.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('./Model_Keras_Weights_summary.h5')
'''
    
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,shuffle=False,\
          batch_size=32,epochs=1,\
          validation_data=(x_val,y_val),\
          callbacks=[reduce_lr,prob_logger,histories] )

model_json = model.to_json()
with open("model_summary.json", "w") as json_file:
     json_file.write(model_json)

model.save_weights('./Model_Keras_Weights_summary.h5',overwrite=True)

