#this file cleans the data.
#Removes all rows that have nan values.
#Writes strings to a text file on which 
#gensim word2vec or corpus creator can be called.
import pandas as pd
import sys
import numpy as np
from generate_features import *
import re
from collections import defaultdict
import gensim
from gensim.models import Word2Vec

text_rank=defaultdict(int)
summary_rank=defaultdict(int)

text_count=defaultdict(int)
summary_count=defaultdict(int)
tproc=text_processor()

summary_tokens=[]
text_tokens=[]

summary_vectors=[]
text_vectors=[]

def Generate_WordVectors():
    global summary_tokens,text_tokens
    print("Start of word2vec:")
    documents=summary_tokens+text_tokens
    start=time.time()
    model=gensim.models.Word2Vec(documents,min_count=1,size=50,workers=4)
    print('Time to generate word vectors: ',time.time()-start)
    model.save('wordvec.wv')
    print(model['good'],model['delight']) 


#this maps words:indices and indices:words
class Mapper():
      def __init__(self,count,index):
          self.Word2Idx=index
          self.WordCount=count
          self.Idx2Word=self.invert_dict()

      def invert_dict(self): 
          d=[(value,key) for key,value in self.Word2Idx.items()] 
          return dict(d)
           
      def idxToWord(self,idx):
          #idx - word index
          #index - dictionary with index:word mapping
          return self.Idx2Word[idx]            

#convert lists of lists of word to lists of lists of intergers - each representing 
#the index of word in dict if its count above threshold else index=0
def tokens_to_vectors(tokens,count,index,max_freq=500):
    #inputs- tokens : lists of lists of words which need mapping to integer values.
    #count - dict with word counts
    #index - dict with word indices
    #max_freq - cut-off freq below which all mapped to 0.
    map_value=0
    vectors=[] 
    def token_mapper(tok_list):
        vector=[]
        for tok in tok_list:
            if(tok=='' or tok ==' '):
              continue
            if(count[tok]>max_freq):
              vector.append(index[tok])
            else:
              vector.append(map_value)
        return vector

    for tok_list in tokens: 
        vectors.append(token_mapper(tok_list))  

    return vectors    

def clean_data(filename='./Reviews.csv'):
    df=pd.read_csv(filename)
    print("original shape: ",df.shape)
    df=df.dropna(axis=0,how='any') 
    print("changed shape: ",df.shape)
    return df

def update_dictionary_text(tokens):
    global text_count,text_rank
    for tok in tokens:
        text_count[tok]+=1  

    for tok in tokens:
        if(text_rank[tok]==0):
           text_rank[tok]=len(text_rank.keys())+1  

def update_dictionary_summary(tokens):
    global summary_rank,summary_count
    for tok in tokens:
        summary_count[tok]+=1 
 
    for tok in tokens:
        if(summary_rank[tok]==0):
           summary_rank[tok]=len(summary_rank.keys())+1  

def write_summary_to_text(df):
    summary=df['Summary'].tolist()
    with open('summary_file.txt','a') as fd:
         for line in summary:
             line=tproc.preprocess_text(line) 
             tokens=tproc.tokenize_text(line)
             print(tokens)
             summary_tokens.append(tokens)
             update_dictionary_summary(tokens) 
             fd.write(line)
             fd.write("\n")
     
def write_Text_to_text(df):
    text=df['Text'].tolist()
    with open('text_file.txt','a') as fd:
         for line in text:
             line=tproc.preprocess_text(line) 
             tokens=tproc.tokenize_text(line)
             tokens=tproc.remove_commonwords_stem(tokens)      
             print(tokens) 
             text_tokens.append(tokens)
             update_dictionary_text(tokens)
             fd.write(line)
             fd.write("\n")

def pickle_vectors(df):
    global summary_tokens,text_tokens,text_count,summary_count,summary_rank,text_rank
    print(df.columns)
    scores=df['Score'].tolist()
    p=dict(Scores=scores,Summary_tokens=summary_tokens,Text_tokens=text_tokens,Text_count=text_count,\
              Summary_count=summary_count,Summary_rank=summary_rank,Text_rank=text_rank)

    with open('dataset2.p','wb') as fp:
         pickle.dump(p,fp) 
    with open('dataset2.p','rb') as fp:
         p1=pickle.load(fp)

if __name__ =='__main__': 
   df=clean_data()
   write_summary_to_text(df)
   write_Text_to_text(df)
   #tokens_to_vectors()
   pickle_vectors(df)
   #Generate_WordVectors()

