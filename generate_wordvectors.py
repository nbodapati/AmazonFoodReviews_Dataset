import re
import time
import os
import csv
from collections import defaultdict
import numpy as np
import sys
import pandas as pd
import gensim
from gensim.models import Word2Vec
removelist="/-"

class sentence_generator():
    #this class acts a generator yielding list of tokens 
    #initiated with a list of sentences.
    def __init__(self,list_of_doc):
       self.list_of_doc=list_of_doc;
    def __iter__(self):    
       for sentence in self.list_of_doc:
           #Remove all caps/named entities before doing this?
           #Construct a generator-2 module.
           sentence=sentence.lower()
           sentence=re.sub(r'<br>|<br />',' ',sentence)
           sentence=re.sub(r'[^A-Za-z0-9\s]','',sentence)
           print(sentence)
           tokens=sentence.split() +['<eos>']
        
           yield tokens


def Generate_WordVectors(directory,filename):
    num_files=0
    list_of_doc=[]
          
    with open(os.path.join(directory, filename), 'r') as readFile:
         #read the entire content within a file into one string.
         sentence=readFile.read()
         list_of_doc.append(sentence) 
  
    print("Start of word2vec:")
    documents=sentence_generator(list_of_doc);
    start=time.time()
    #generate a word vector for any word that has occured for 
    #more than min_count times.
    model=gensim.models.Word2Vec(documents,min_count=1,size=200,workers=4)
    print('Time to generate word vectors: ',time.time()-start)
    model.save('amazon_foodreviews.wv')
    


def main():
    if(os.name =="nt"): 
       dataDirectory = os.getcwd() 
    elif(os.name=="posix"):
       dataDirectory = os.getcwd() + "/"
       #reload(sys)
       #sys.setdefaultencoding('latin-1')
  
    
    else:
         print("Unknown OS")
    Generate_WordVectors(dataDirectory,'text_file.txt')

if __name__ == '__main__':
    main()

