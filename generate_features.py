import sys
import pandas as pd
import gensim
from gensim.models import Word2Vec
import numpy as np
import time
import os
import pickle
import re
import json
from stemming.porter2 import stem

class text_processor(object):
      def __init__(self):
         self.__name__='Text processing library'
         self.removelist=''

      def remove_commonwords_stem(self,token_list):
          tokens=[stem(tok) for tok in token_list if len(tok)>3 and\
                                       len(tok)<10]
          return tokens
      
      def remove_commonwords(self,token_list):
          tokens=[tok for tok in token_list if len(tok)>3]
          return tokens

      def to_lower(self,sentence):
          #Turn the sentence to lower characters.
          return sentence.lower()

      def preprocess_text(self,sentence):
          #Takes an sentence, makes it all lower, removes special char.
          sentence=self.to_lower(sentence)
          sentence=re.sub(r'<br>|<br />|\n',' ',sentence)
          sentence=re.sub(r'[^A-Za-z0-9\s'+self.removelist+']',' ',sentence)
          sentence=re.sub(r'[\' \']+',' ',sentence)
          print(sentence)
          print()
          return sentence

      def sentence_stream(self,text_file='./text_file.txt'):
          #This function takes in as input a text file name
          #Returns one sentence each time it gets called.
          with open(text_file,'rb') as f:
               while(1):
                  line=f.readline()
                  if(line==''):
                    break
                  #pre-process,tokenize the sentence and send back.
                  line=line.decode('utf-8')
                  line=self.preprocess_text(line)
                  tokens=self.tokenize_text(line)
                  tokens=self.remove_commonwords(tokens)
                  yield tokens

      def tokenize_text(self,sentence,delimiter_=' '):
          #take as input sentence,a delimiter to split on.
          return sentence.split(delimiter_)+['<eos>']

class word2vec_processing():
    def __init__(self,directory='./',filename='amazon_foodreviews.wv',data=None):
        self.file=os.path.join(directory,filename)
        self.model=None
        self.data=data #dataset to feed NxTxd

    def load_file(self):
        start=time.time()
        self.model=gensim.models.Word2Vec.load(self.file)
        print('time to load wordvectors: ',time.time() - start)

    def create_dataset(self,max_tokens=100):
        #use model to find each word rep and add to data.
        textproc=text_processor()
        sample_number=0
        self.load_file()
        for tokens in textproc.sentence_stream():
            if(sample_number==self.data.shape[0]):
              break 

            time_step=0
            for tn,token in enumerate(tokens):
                print(sample_number)
                if(tn==max_tokens):
                  break 

                if(token==''):
                  continue  
                rep=self.model[token]
                self.data[sample_number,time_step,:]=rep
                print(self.data[sample_number,time_step,:])  
                time_step+=1
            sample_number+=1 #increment with each sentence

        return self.data

if __name__ =='__main__':
    #loop over each sentence and for each word find its 200-dimensional
    #word vector representation and make an array of
    #(NxTxd) where N is the number of samples,
    #T is the max number of time steps,
    #d is the dimensionality of word vectors.
    d=200
    corpus=pickle.load(open("amazon_text_corpus.p", "rb"))
    N=len(corpus) //16  #only using the first quarter.
    corpus=corpus[:N]
    T =len(sorted(corpus,key=len, reverse=True)[0])
    print(N,T,N*T*d)
    data=np.zeros((N,T,d))
    print(data.shape) 
    w2v_proc=word2vec_processing(data=data)
    data=w2v_proc.create_dataset(max_tokens=T)
    #json.dump(data.tolist(),open('corpus_w2vrep.p','wb'))
    pickle.dump(data,open('corpus_w2vrep_large.p','wb'))
    #Pickle it to file corpus_w2vrep.p
