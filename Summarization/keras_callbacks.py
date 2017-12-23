import keras
import pickle

class Histories(keras.callbacks.Callback):
    def __init__(self,test_data,pkl,model_pkl,\
                 weights_pkl):
        self.test_data=test_data
        self.pkl=pkl
        self.model_pkl=model_pkl
        self.weights_pkl=weights_pkl

    def on_train_begin(self, logs={}):
        self.epoch_losses = []
        self.batch_losses = []
        print("Running callbacks..")

    def on_train_end(self, logs={}):
        #write the loses to a pickle 
        print("Ending callbacks.. with pickling")
        pickle.dump(dict(epoch_losses=self.epoch_losses,batch_losses=self.batch_losses),open(self.pkl,'wb'))
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        #logs has acc,loss,val_loss etc.
        self.epoch_losses.append(logs.get('loss'))
        #How to get predictions at the end of each epoch. 
        '''
        x,y=self.test_data #has x_val,y_val
        y_pred = self.model.predict(x)
        #check how many are class -0 to class -1
        print("x_pred: ",x)
        print("y_pred:",y_pred)
        print("y_actual: ",y)
        '''
        model_json = self.model.to_json()
        with open(self.model_pkl, "w") as json_file:
             json_file.write(model_json)
        self.model.save_weights(self.weights_pkl,overwrite=True)

        
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        self.batch_losses.append(logs.get('loss'))
        return
