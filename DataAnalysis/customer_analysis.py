#Author:Nagasravika Bodapati
import pandas as pd
import numpy as np
import math
from collections import defaultdict
import pickle
import time

  

def clean_names(name):
    if(type(name)!=str):
       return name

    if("\"" in name):
       names=name.split("\"")
       return names[0]
    else:
       return name

def rows_to_str(row):
    #example: [1,1,1,1] ==> '1111'
    row=[str(int(r)) for r in row]
    return ''.join(row)

def str_to_bytes(string):
    #example: '1111'=>15
    return int(string,2)

def create_defaultdict():
    d=defaultdict(list)
    return d

Reviews_df=pd.read_csv('./Reviews.csv')
Reviews_df=Reviews_df.dropna(axis=0, how='any')
Reviews_df=Reviews_df.reset_index()
print(Reviews_df.head())

Reviews_df['ProfileName']=list(map(clean_names,Reviews_df['ProfileName']))
#Clean the names by removing pet names
customers=Reviews_df['ProfileName'].unique()
products=Reviews_df['ProductId'].unique()
print("Num customers: ",len(customers))

grouped_customers=Reviews_df.groupby(['ProfileName'])
def get_products(customer):
    global grouped_customers,products
    customer_df=grouped_customers.get_group(customer)
    products_=customer_df['ProductId'].unique()

    def is_sampled(product):
        return (product in products)
    #print(products_)
    products_=list(filter(is_sampled,products_))
    return products_

count=0
def counter(func):
    def function_wrapper(x):
        global count
        #print("Before calling " + func.__name__,count)
        print("Customer_number ",count)
        count+=1
        return func(x)
        #print("After calling " + func.__name__)
    return function_wrapper

#equivalent to: big_customer=counter(big_customer)   
@counter
def big_customer(customer):
    n_products=len(get_products(customer))
    print(customer,n_products>30)
    return (n_products>30)

#use a sampled subset of customers and products -every other.
#this is why min-hashing
customers1=customers[::10]
customers=list(filter(big_customer,customers))
pickle.dump(customers,open('filtered_customers.pkl','wb'))
products=products
print(len(customers),len(products))


def print_dict(d):
    for k,v in d.items():
        print("Key: ",k,"Value: ",v)

def sparse_matrix_handler(matrix):
    global customers,products
    non_zero=[]
    product_customers=[]
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if(matrix[row,col]==1):
               non_zero.append((row,col))
               product_customers.append([products[row],customers[col]])
    return (non_zero,product_customers)


matrix=np.zeros((len(products),len(customers)))

pc_df=pd.DataFrame(matrix,columns=customers,index=products)
print("Starting building matrix..")
start=time.time()
for customer in customers:
    print(customer)
    products_=get_products(customer)
    #the products are checked to see if they exist as columns or not.
    pc_df.loc[products_,customer]=1

print("Time to build matrix: ",time.time()-start)
pc_matrix=pc_df.as_matrix()
print(np.sum(pc_matrix,axis=1))
#nz,pc=sparse_matrix_store(pc_matrix)
#print(pc_df.head(),nz,pc)

#create 10 bands to represent total products
num_bands=100
r=len(products)//num_bands
num_buckets=math.pow(2,r)-1
print("Num rows in each band: ",r)
print("Num buckets in each band: ",num_buckets)

band_buckets=[]
for i in range(num_bands+1):
    band_buckets.append(create_defaultdict())
 
#turn the df back to numpy array 
print("Starting bucket filling with bands...")
start=time.time()

for col in range(pc_matrix.shape[1]):
    pc_col=pc_matrix[:,col].reshape(-1,1)
    band_num=0
    for row_start in range(0,pc_matrix.shape[0],r):
        #progresses as 0,10,20,30
        band=(pc_col[row_start:row_start+r]).flatten().tolist()
        band=rows_to_str(band)
        #print("Band: ",band)
        band=str_to_bytes(band)
        #print("Band: ",band,band_num)
        #band_buckets[band_num] => defaultdict
        band_buckets[band_num][band].append(customers[col]) 
        band_num+=1
print("Time to build band_buckets: ",time.time()-start)

for band_num in range(num_bands+1):
    print("Band num: ",band_num)
    print_dict(band_buckets[band_num])
pickle.dump(band_buckets,open('band_buckets.pkl','wb'))

