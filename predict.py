import keras 
from keras.models import load_model

import pickle
import pandas as pd
import numpy as np

import math, random
import os,sys
import time
from keras import backend as K
from model_bkup import *

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


def load_data():
    data=pickle.load(open("dataset.p", "rb"))
    corpus=process_data(data)

    target=np.array(data['Scores']).reshape(-1,1)
    corpus=corpus['summary']
    #corpus=corpus['text']
    target_=target-1
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
def train_val_test_split(train_percent=0.15,val_percent=0.01,test_percent=0.6,data_=None):
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

class prediction():
      def predict(self):
          data,weights=load_data()
          split,n_train=train_val_test_split(data_=data)

          self.x_train=split['x_train']
          self.x_val=split['x_val']
          self.x_test=split['x_test']

          self.y_train=split['y_train']
          self.y_val=split['y_val']
          self.y_test=split['y_test']
          print(self.x_train.shape,self.x_val.shape,self.x_test.shape,self.y_train.shape,self.y_val.shape)

          self.model=load_model('./Model_Keras_large.h5')
          self.model.load_weights('./Model_Keras_Weights_large.h5')
           
          y_pred=self.model.predict(self.x_val,batch_size=32)
          pred_labels=np.argmax(y_pred,axis=1)
          actual_pred=np.argmax(self.y_val,axis=1)
          print(pred_labels)
          print(actual_pred)
          print("Accuracy: ",np.mean(np.array(pred_labels)==np.array(actual_pred)))

      def get_activations(self,print_shape_only=False, layer_name=None):
          print('----- activations -----')
          activations = []
          inp = self.model.input
          model_multi_inputs_cond = True
          model_inputs=self.x_val
 
          if not isinstance(inp, list):
          # only one input! let's wrap it in a list.
             inp = [inp]
             model_multi_inputs_cond = False
             outputs = [layer.output for layer in self.model.layers if
             layer.name == layer_name or layer_name is None]  # all layer outputs
             funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

             if model_multi_inputs_cond:
                list_inputs = []
                list_inputs.extend(model_inputs)
                list_inputs.append(0.)
             else:
                list_inputs = [model_inputs, 0.]
                # Learning phase. 0 = Test mode (no dropout or batch normalization)
                # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]

          layer_outputs = [func(list_inputs)[0] for func in funcs]
          for layer_activations in layer_outputs:
               activations.append(layer_activations)

          if print_shape_only:
             print(layer_activations.shape)
          else:
             print(layer_activations[-1])

          
pred=prediction()
pred.predict()
pred.get_activations()
