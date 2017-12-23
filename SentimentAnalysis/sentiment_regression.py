from keras.models import Sequential
from keras.layers import Dense,Dropout,Masking,TimeDistributed
from keras.layers import Embedding 
from keras.layers import LSTM
import keras 
from keras.models import load_model

import pickle
import pandas as pd
import numpy as np

import math, random
import os,sys
import time
import keras.backend as K

from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence

from model_bkup import *
from keras_callbacks import *
from keras.models import Model
from keras.models import model_from_json

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score

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
    data=dict(data=corpus,labels=target)
    return data

#split data and targets into train,test,val sets.
def train_val_test_split(train_percent=0.1,val_percent=0.008,test_percent=0.6,data_=None):
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

def predict(x_train,y_train,\
            x_val,y_val,\
            x_test,y_test,model,layer_num1=3,layer_num2=4):

    from keras.models import model_from_json
    import json
    json_file = open('model_large_reg.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./Model_Keras_Weights_large_reg.h5")
    print("Loaded model from disk")

    lstm_layer_model = Model(inputs=loaded_model.input,
                                 outputs=loaded_model.get_layer(index=layer_num1).output)
    lstm_output = lstm_layer_model.predict(x_val)
    print("Lstm output: ",lstm_output,lstm_output.shape)   

    lstm_layer_model = Model(inputs=loaded_model.input,
                                 outputs=loaded_model.get_layer(index=layer_num2).output)
    lstm_output2 = lstm_layer_model.predict(x_val)

    y_pred=lstm_output2
    y_val1=y_val
    pickle.dump(dict(lstm_output=lstm_output.tolist(),y_pred=list(y_pred),y_val=list(y_val1)),\
                    open('output_large_reg.pkl','wb'))
         
    loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop')
    score = loaded_model.evaluate(x_val, y_val, verbose=0)
    print(score)



data=load_data()
split,n_train=train_val_test_split(data_=data)
#callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=2, min_lr=0.001)

prob_logger=keras.callbacks.ProgbarLogger(count_mode='samples')


x_train=split['x_train']
x_val=split['x_val']
x_test=split['x_test']

y_train=split['y_train']
y_val=split['y_val']
y_test=split['y_test']

time_steps=x_train.shape[1]
num_features=1
histories=Histories((x_val,y_val),'Large_Reg.pkl',\
          "model_large_reg.json",'./Model_Keras_Weights_large_reg.h5')

#USe tensorboard to viz the results.
tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs', write_graph=True)

print(x_train.shape,x_val.shape,x_test.shape,y_train.shape,y_val.shape)

def neural_net(): 
    global max_features
    model=Sequential()
    model.add(Embedding(max_features,128))
    model.add(Masking(mask_value=0.0,input_shape=(time_steps,num_features)))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2)) #hidden memory unit output is a sequence of 128 float values
    model.add(Dense(1))
    return model

model=neural_net()
json_file = open('model_large_reg.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('./Model_Keras_Weights_large_reg.h5')

model.compile(loss='mean_squared_error',
              optimizer='rmsprop')

model.fit(x_train,y_train,shuffle=False,\
          nb_epoch=10,batch_size=32,verbose=1,\
          validation_data=(x_val,y_val),callbacks=[early_stopping,histories])
'''
estimator = KerasRegressor(build_fn=neural_net, nb_epoch=5, batch_size=32, verbose=1)
estimator.fit(x_train,y_train,shuffle=True)
'''
score = model.evaluate(x_val, y_val, batch_size=16)
print("Score:",score)

model_json = model.to_json()
with open("model_large_reg.json", "w") as json_file:
     json_file.write(model_json)

model.save_weights('./Model_Keras_Weights_large_reg.h5',overwrite=True)

predict(x_train,y_train,\
            x_val,y_val,\
            x_test,y_test,model,layer_num1=3,layer_num2=4)


