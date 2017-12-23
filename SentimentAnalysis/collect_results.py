import keras 
from keras.models import load_model

import pandas as pd
import numpy as np

from keras.models import Model
from keras.models import model_from_json
import json
import pickle
def predict(split,model,output_file='output_summary.pkl',model_file='model_summary.json',\
                 weights_file='./Model_Keras_Weights_summary.h5',**layer_names):
 
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights(weights_file)
    print("Loaded model from disk")

    x_val=split['x_val']
    y_val=split['y_val']
    y_val2=split['y_val2']
    text_val=split['text_val']
    summary_val=split['summary_val']
    
    lstm_layer_model = Model(inputs=loaded_model.input,
                                 outputs=loaded_model.get_layer(layer_names['layer1']).output)
    lstm_output = lstm_layer_model.predict([x_val,y_val2])
    print("Lstm output: ",lstm_output,lstm_output)   

    lstm_layer_model = Model(inputs=loaded_model.input,
                                 outputs=loaded_model.get_layer(layer_names['layer2']).output)

    lstm_output2 = lstm_layer_model.predict([x_val,y_val2])
    y_pred=np.argmax(lstm_output2,axis=2)
    y_pred=y_pred.reshape(*y_pred.shape,1)
    print(len(lstm_output2),lstm_output2.shape)
    
    pickle.dump(dict(text_tokens=text_val,summary_tokens=summary_val,\
                     lstm_output=lstm_output,y_pred=y_pred,y_val=y_val2),\
                     open(output_file,'wb'))
          
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = loaded_model.evaluate([x_val,y_val2], y_val, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


