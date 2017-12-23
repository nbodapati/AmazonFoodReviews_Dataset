#1)Read the outputs and map indices to words
#2)Train an encoder-decoder for low dimensional representation of 
#output vectors from layers.

import pickle
import numpy as np
import pandas as pd
from cleaning_summary import Mapper


class Interpret():
      def __init__(self,pkl):
          self.pkl=pkl
          self.load_dicts()
          self.output_file='./summarization_results.txt'
          self.fd=open(self.output_file,'w')
     
      def load_dicts(self):
          self.data=pickle.load(open("dataset2.p", "rb"))
          #keys: Scores,Summary_tokens,Text_tokens
          #Text_count,Summary_count,Text_rank,Summary_rank
          self.desc_mapper=Mapper(self.data['Text_count'],self.data['Text_rank'])           
          self.head_mapper=Mapper(self.data['Summary_count'],self.data['Summary_rank'])           

      def print_list(self,lst):
         for l in lst:
             print(l,end=' ',file=self.fd)
         print(file=self.fd)

      def interpret_output(self,arr):
          #arr - of shape (n_timesteps,1)
          words=[]
          for i in range(arr.shape[0]):
              for j in range(arr.shape[1]):
                  idx=arr[i,j]
                  if(idx==0):
                    words.append('<UNK>') 
                    continue
                  words.append(self.head_mapper.idxToWord(idx))
          self.print_list(words) 

      def read_input(self):
          self.results=pickle.load(open(self.pkl,'rb'))
          print(self.results.keys()) 
          #y_pred and y_val2 of shapes (N,15,1) 
          self.y_pred=self.results['y_pred']
          self.y_val=self.results['y_val']
          #print_list(self.y_pred.shape)
          #print_list(self.y_val.shape)
          print(np.mean(self.y_pred==self.y_val)) 
          #interpret first few results
          for i in range(self.y_pred.shape[0]):
              print("y_pred:",end='  ',file=self.fd)
              self.interpret_output(self.y_pred[i,:,:])  
              print("y_val:",end='  ',file=self.fd)
              self.interpret_output(self.y_val[i,:,:])  

if __name__ == '__main__':
   interpret=Interpret('./output_summary.pkl')
   interpret.read_input()    
