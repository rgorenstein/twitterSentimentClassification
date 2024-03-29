"""code written for Python 3"""

"""saveEmbedding.ipynb

# Imports
"""
import gensim
import pandas as pd
import numpy as np
import string
from datetime import datetime
import re

"""# Definitions"""

def saveEmbed(trainContent):
    embed_s = datetime.now()
    model = gensim.models.Word2Vec(trainContent,size=50,window=1,min_count=6,workers=1)
    model.train(trainContent, total_examples=len(trainContent), epochs=200) 
    embed_e = datetime.now()
    embed_time = embed_e - embed_s
    W_embed_dict = dict({})
    for idx, key in enumerate(model.wv.vocab):
        W_embed_dict[key] = model.wv[key]
    W_df = pd.DataFrame(W_embed_dict)
    return(embed_time,W_df)

"""# Load Data

## [Data Source (Kaggle)](https://www.kaggle.com/dataturks/dataset-for-detection-of-cybertrolls)
"""

data_file='cyberTrolls.json'
data = pd.read_json(data_file,lines = True)

annotation = pd.DataFrame(data['annotation'].tolist())
data = data.drop(columns = ['annotation'])
data = pd.concat([data,annotation],axis=1)
label = data['label'].tolist()
labels = []
for x in range(0,20001):
  labels.append(label[x][0])  
labels = [x.encode('UTF8') for x in labels]
labels = pd.DataFrame(labels)
data = data.drop(columns=['extras','label'])
data = pd.concat([data,labels],axis=1)
data.rename(columns={0:'label'}, inplace=True)
content = data['content'].tolist()
content = [x.encode('UTF8')for x in content]
content2 = []
content3 = []
for x in range(0,len(content)):
  content3.append(re.sub('['+string.punctuation+']', '',content[x].decode('UTF8')))
for x in range(0,len(content3)):
  content2.append(content3[x].lower())
full_content_list = []
for x in range(0,len(content2)):
    full_content_list.append(content2[x].split())
for x in range(0,len(content)):
  data['content'][x] = full_content_list[x]
data = data[data.astype(str)['content'] != '[]']

"""## Split Data"""

all_index = range(len(data))

"""obtain test set"""
test_file = open("test_index.txt",'r')
test_index = [int(i.strip('\n')) for i in test_file.readlines()]

"""remove test set from full set"""
index = [x for x in all_index if x not in test_index]
test_list = []
for x in range(0,len(test_index)):
  test_list.append(full_content_list[test_index[x]])
content_list = []
for x in range(0,len(index)):
  content_list.append(full_content_list[index[x]])

"""# Build & Save Word Embedding"""

embed_time,W_df = saveEmbed(content_list)
W_df.to_pickle('embedding.pkl')

f = open('embed_time.txt','w')
f.write(f'{embed_time}')
f.close()
