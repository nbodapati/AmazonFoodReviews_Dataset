#In this implementation,each word in the dictionary is indexed by 
#its frequency of occurence and not its position in the dictionary 
#as is the case with data.py

import pandas as pd
import os,sys
import numpy as np
import re
import torch
from collections import defaultdict
from itertools import islice
import pickle

import time

from mrjob.job import MRJob
from mrjob.step import MRStep


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

data={}
corpus=[]

class Make_Corpus(MRJob):
      def __init__(self,args):
          super(Make_Corpus, self).__init__(args=args)
          self.sentence_num=-1

      def mapper_get_words(self,_,sentence):
          sentence=sentence.lower()
          sentence=re.sub(r'<br>|<br />',' ',sentence)
          sentence=re.sub(r'[^A-Za-z0-9\s]','',sentence)
          tokens=sentence.split() +['<eos>']
          self.sentence_num+=1

          for token in tokens:
              yield(self.sentence_num,data[token])

      def reducer_update_corpus(self,token,values):
          #print(token,values)
          corpus.append(list(values))
          yield ('updated','updated')

      def steps(self):
          return [
            MRStep(mapper=self.mapper_get_words,
                   reducer=self.reducer_update_corpus)
            ] 

class Make_Dictionary(MRJob):
      def __init__(self,args):
          super(Make_Dictionary, self).__init__(args=args)
          self.sentence_num=-1

      def mapper_get_words(self,_,sentence):
          sentence=sentence.lower()
          sentence=re.sub(r'<br>|<br />',' ',sentence)
          sentence=re.sub(r'[^A-Za-z0-9\s]','',sentence)
          print("Input sentence: ",sentence)
          tokens=sentence.split() +['<eos>']
          self.sentence_num+=1
          #print(self.sentence_num)
          for token in tokens:
              yield(token,1)

      def reducer_update_dict(self,token,values):
          #print(token,values)
          data[token]=sum(values)
          yield (token,data[token])

      def steps(self):
          return [
            MRStep(mapper=self.mapper_get_words,
                   reducer=self.reducer_update_dict),
        ] 
       
if __name__ =='__main__':
   start=time.time()
   Make_Dictionary(sys.argv[1]).run()
   print("Time to build dictionary: ",time.time() - start)
   #write the dictionary to a file
   print(take(100,data.items()))
   Make_Corpus(sys.argv[1]).run() 
   pickle.dump(corpus,open("amazon_text_corpus2.p", "wb" ))    
 


