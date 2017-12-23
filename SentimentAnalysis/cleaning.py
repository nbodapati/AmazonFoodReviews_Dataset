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

word_count=defaultdict(int)
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
    #generate a word vector for any word that has occured for 
    #more than min_count times.
    #dimensionality - lower, poorer performance. 
    model=gensim.models.Word2Vec(documents,min_count=1,size=50,workers=4)
    print('Time to generate word vectors: ',time.time()-start)
    model.save('wordvec.wv')
    print(model['good'],model['delight']) 

def map_tokens(token):
    global word_count
    return word_count[token]


def tokens_to_vectors():
    global summary_tokens,text_tokens
    for tok_list in summary_tokens:
        summary_vectors.append(list(map(map_tokens,tok_list)))  
    for tok_list in text_tokens:
        text_vectors.append(list(map(map_tokens,tok_list)))  
        

def clean_data(filename='./Reviews.csv'):
    df=pd.read_csv(filename)
    print("original shape: ",df.shape)
    df=df.dropna(axis=0,how='any') 
    print("changed shape: ",df.shape)
    return df

def update_dictionary(tokens):
    global word_count
    for tok in tokens:
        word_count[tok]+=1  

def write_summary_to_text(df):
    summary=df['Summary'].tolist()
    with open('summary_file.txt','a') as fd:
         for line in summary:
             line=tproc.preprocess_text(line) 
             tokens=tproc.tokenize_text(line)
             tokens=tproc.remove_commonwords_stem(tokens)     
             print(tokens)
             summary_tokens.append(tokens)
             update_dictionary(tokens) 
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
             update_dictionary(tokens)
             fd.write(line)
             fd.write("\n")

def pickle_vectors(df):
    global summary_vectors,text_vectors,word_count
    print(df.columns)
    scores=df['Score'].tolist()
    p=dict(Scores=scores,Summary_vectors=summary_vectors,Text_vectors=text_vectors,Word_count=word_count)
    with open('dataset2.p','wb') as fp:
         pickle.dump(p,fp) 
    with open('dataset2.p','rb') as fp:
         p1=pickle.load(fp)
    #print(p1['Word_count'])

df=clean_data()
write_summary_to_text(df)
write_Text_to_text(df)

tokens_to_vectors()
pickle_vectors(df)
Generate_WordVectors()
