import pandas as pd
import os,sys
import numpy as np
import re
import torch
from collections import defaultdict
from itertools import islice
import pickle

Reviews_df=pd.read_csv('./Reviews.csv')
text=Reviews_df['Text'].tolist()
print('Num text reviews:',len(text))


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

class Dataset(object):
    def __init__(self):
        self.token_list=[]
        self.dict_words={}
        self.num_tokens=0
        self.corpus=self.get_corpus()

    def add_word(self,token):
        if token not in self.token_list:
            if(token=='dont'):
                print(token)
            self.token_list.append(token)

    def tokenize_doc(self):
        #Build a list/set with unique words in the text.
        sent_num=0
        for sentence in text:
            #print(sentence)
            sentence=sentence.lower()
            sentence=re.sub(r'<br>|<br />',' ',sentence)
            sentence=re.sub(r'[^A-Za-z0-9\s]','',sentence)
            #print(sent_num)
            sent_num+=1
            tokens=sentence.split() +['<eos>']
            for token in tokens:
                self.add_word(token)

        #build dictionary using the list of unique tokens generated above.
        self.num_tokens=len(self.token_list)
        self.dict_words=dict(zip(self.token_list,list(range(self.num_tokens))))
        print(take(10,self.dict_words.items()))
        print("Num tokens: ",self.num_tokens)

    def generate_dataset(self):
        corpus=[]
        for sentence in text:
            sentence=sentence.lower()
            sentence=re.sub(r'<br>|<br />',' ',sentence)
            sentence=re.sub(r'[^A-Za-z0-9\s]','',sentence)
            tokens=sentence.split() +['<eos>']
            sentence_ids=[]
            for token in tokens:
                #if(self.dict_words[token])
                sentence_ids.append(self.dict_words[token])

            corpus.append(sentence_ids)

        return corpus

    def get_corpus(self):
        self.tokenize_doc()
        corpus=self.generate_dataset()
        return corpus

corpus=Dataset()
corpus=corpus.corpus
pickle.dump(corpus,open("amazon_text_corpus.p", "wb" ))
