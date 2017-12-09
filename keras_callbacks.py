import keras


class Histories(keras.callbacks.Callback):
    def __init__(self,test_data):
        self.test_data=test_data

    def on_train_begin(self, logs={}):
        self.losses = []
        print("Running callbacks..")
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        #logs has acc,loss,val_loss etc.
        self.losses.append(logs.get('loss'))
        x,y=self.test_data #has x_val,y_val
        y_pred = self.model.predict(x)
        #check how many are class -0 to class -1
        print("x_pred: ",x)
        print("y_pred:",y_pred)
        print("y_actual: ",y)
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return
