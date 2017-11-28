import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import math, random
import os,sys
import time

#load the corpus using pickle
corpus=pickle.load(open( "amazon_text_corpus.p", "rb" ))
print("Corpus length: ",len(corpus))
print(corpus[:3])
N=len(corpus)//4 # only using the first quarter.
corpus=corpus[:N]

length = len(sorted(corpus,key=len, reverse=True)[0])
seq_lengths=[len(xi) for xi in corpus]
print(len(seq_lengths))

c=[]

t=time.time()
for x in corpus:
    c.append(x+[0]*(length-len(x)))
print("t= ",time.time()-t)
'''
t=time.time()
for x in corpus[N:]:
    c.append(x+[0]*(length-len(x)))
print("t= ",time.time()-t)
'''

corpus=np.array(c)
corpus_df=pd.DataFrame(corpus)
corpus_df['seq_lengths']=seq_lengths
print(corpus_df.head())

corpus_df.sort_values(by='seq_lengths',ascending=False,axis=0,inplace=True)
print(corpus_df.head())

corpus=np.array(corpus_df,dtype=float)
seq_lengths=corpus[:,-1]
corpus=corpus[:,:-1]

print(corpus.shape)
batch_size = 3
max_length = length
hidden_size = 10
n_layers =1
num_input_features=1

batch_in = torch.zeros((batch_size,num_input_features, max_length))
print("Size of batch_in: ",batch_in.size())

#batch_in=torch.LongTensor(corpus[:batch_size])
for  x in range(batch_size):
    batch_in[x]=torch.FloatTensor(corpus[x,:])

batch_in = Variable(batch_in)

pack = torch.nn.utils.rnn.pack_padded_sequence(batch_in, seq_lengths[:batch_size], batch_first=True)
#initialize
rnn = nn.RNN(max_length, hidden_size, n_layers, batch_first=True)
h0 = Variable(torch.randn(n_layers, batch_size, hidden_size).float())

#forward
out, hn = rnn(pack, h0)
# unpack
unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out)
print(unpacked)
print(hn)

#print(rnn.weight_ih_l[0])
