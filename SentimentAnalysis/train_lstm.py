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
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.models import model_from_json

max_features=None

def process_data(data):
    global max_features
    word_count=data['Word_count']
    summary_vectors=data['Summary_vectors']    
    text_vectors=data['Text_vectors']    
    sum_length =10 #len(sorted(summary_vectors,key=len, reverse=True)[0])
    text_length=500#len(sorted(text_vectors,key=len, reverse=True)[0]) 
    
    n_samples=len(summary_vectors)
    summary_=np.zeros((n_samples,sum_length,1))
    text_=np.zeros((n_samples,text_length,1))
    summary_vectors=sequence.pad_sequences(summary_vectors,maxlen=sum_length) 
    text_vectors=sequence.pad_sequences(text_vectors,maxlen=text_length) 
    max_features=np.max(summary_vectors)+1
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
    corpus=corpus['summary']
    #corpus=corpus['text']
    target_=target-1

    data_=shuffle_data(data_=dict(data=corpus,labels=target))
    corpus=data_['data']
    target=data_['labels'] 
    print(corpus.shape) 

    n_samples=(corpus.shape[0])
    n_steps=corpus.shape[1] 
    weights=0 

    #convert to one-hot encoding.
    target=keras.utils.to_categorical(target_,num_classes=5)
    data=dict(data=corpus,labels=target)
    print(data.keys()) 
    return data,weights

#split data and targets into train,test,val sets.
def train_val_test_split(train_percent=0.3,val_percent=0.008,test_percent=0.6,data_=None):
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
            x_test,y_test,model,layer_num=3):

    from keras.models import model_from_json
    import json
    json_file = open('model_small.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./Model_Keras_Weights_small.h5")
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
    print(y_pred)
    print(y_val1)
    pickle.dump(dict(lstm_output=lstm_output.tolist(),y_pred=list(y_pred),y_val=list(y_val1)),\
                    open('output_small.pkl','wb'))
         
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = loaded_model.evaluate(x_val, y_val, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))



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
histories=Histories((x_val,y_val),'loses_small.pkl','model_small.json',\
                 './Model_Keras_Weights_small.h5')

print(x_train.shape,x_val.shape,x_test.shape,y_train.shape,y_val.shape)

model=Sequential()
model.add(Embedding(max_features,128))
model.add(Masking(mask_value=0.0,input_shape=(time_steps,num_features)))

model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2)) #hidden memory unit output is a sequence of 128 float values
model.add(Dense(5,activation='softmax'))

rmsprop=keras.optimizers.rmsprop(lr=1e-5)#,decay=0.01)
sgd=keras.optimizers.adam(lr=10,decay=0.01)

#model=load_model('./Model_Keras_large.h5')
json_file = open('model_small.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('./Model_Keras_Weights_small.h5')
print("Parallelizing model..")
#parallel_model = multi_gpu_model(model, gpus=4)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train,y_train,shuffle=False,\
          batch_size=128,epochs=10,\
          validation_data=(x_val,y_val),\
          callbacks=[reduce_lr,prob_logger,histories] )
#model.save('./Model_Keras_large.h5',overwrite=True)
model_json = model.to_json()
with open("model_small.json", "w") as json_file:
     json_file.write(model_json)

model.save_weights('./Model_Keras_Weights_small.h5',overwrite=True)

predict(x_train,y_train,\
            x_val,y_val,\
            x_test,y_test,model,layer_num=3)

