import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random

import pandas as pd
import os,sys


Reviews_df=pd.read_csv('./Reviews.csv')
print(Reviews_df.columns)

scores=Reviews_df['Score'].tolist()
print(scores[:5])
text=Reviews_df['Text'].tolist()
print(text[:5])

#set torch seed for reproducibility
torch.manual_seed(1111)
print(torch.cuda.is_available()) #true


