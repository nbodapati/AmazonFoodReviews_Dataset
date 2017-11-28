import pandas as pd
import sys

filename=sys.argv[1]
Reviews_df=pd.read_csv(filename)
text=Reviews_df['Text'].tolist()
with open('text_file.txt','a') as fd:
     for line in text:
         fd.write(line)
         fd.write("\n")
          



