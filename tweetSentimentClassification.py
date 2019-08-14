"""tweetSentimentClassification.ipynb

# Imports and Definitions

## Imports
"""
import pandas as pd
import numpy as np
import string
from datetime import datetime
import re
import collections
from scipy import sparse
import math
from bisect import bisect_left
from sklearn.metrics.pairwise import euclidean_distances
from pyemd import emd

"""## Definitions

### `intersect`, `union`, and `jaccard`
"""

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))
  
def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))
  
def jaccard(a,b):
  """ return the Jaccard distance btwn two lists """
  return 1 - (float(len(intersect(a,b)))/len(union(a,b)))

"""### `BOWPrep`"""

class BOWPrep:
  """
  Parameters
  ----------

  vocabList : list, size: vocab_size
      List of vocabulary obtained from embedding.
      
  vocabDictionary : dict, size: vocab_size
  
  embeddingDictionary : dict, size: vocab_size
      Dictionary of vocabulary items (keys) and their corresponding precomputed word embeddings (values).
      
  W_embed : array, shape: (vocab_size, embed_size)
      Precomputed word embeddings between vocabulary items.
      Row indices should correspond to the columns in the bag-of-words input.

  """
  def __init__(self, vocabList, vocabDictionary, embeddingDictionary, W_embed):
    self.vocabList = vocabList
    self.vocabDict = vocabDictionary
    self.embedDict = embeddingDictionary
    self.embedding = W_embed

  def removeVocab(self, tweet): 
    """ each tweet is a list of the individual word strings """
    toRemove = []
    for w in tweet:
      if w not in self.vocabList:
        toRemove.append(w)
    for x in toRemove:
      tweet.remove(x)
    return(tweet)
  
  def _removeEmpty(self, data, labels):
    empty = []
    for x in range(0,len(data)):
      if len(data[x]) == 0:
        empty.append(x)
    return(list(np.delete(np.array(data),np.array(empty))),np.delete(np.array(labels),np.array(empty)))
    
  def _dataPrep(self, Data, labels):
    data = self._removeEmpty(Data,labels)[0]
    wordCounts = {}
    for x in range(0,len(data)):
      wordCounts[x] = dict(collections.Counter(data[x]))
    wc_keys = list(wordCounts.keys())
    bow_frac = {}
    for x in range(0,len(wc_keys)):
      bow_frac[wc_keys[x]] = np.divide(list(np.array(sorted(wordCounts[x].items()))[:,1].astype(int)),sum(wordCounts[x].values()))
    bf_keys = []
    for x in range(0,len(data)):
      if x in list(bow_frac.keys()):
        bf_keys.append(x)
    full_bow = []
    bow = [0]*len(self.vocabList)
    for t in bf_keys:
      x = 0
      for w in list(np.array(sorted(wordCounts[t].items()))[:,0].astype(str)):
        try:
          bow[self.vocabDict[w]] = bow_frac[t][x]
        except:
          pass
        x += 1
      full_bow.append(bow)
      bow = [0]*len(self.vocabList)
    return(np.asarray(full_bow,dtype='float64'))
  
  def _makeSparse(self,data,labels):
    notSparse = self._dataPrep(data,labels)
    return(sparse.csr_matrix(notSparse))
  
  
  def _clean(self, data, labels):
    return(self._dataPrep(data, labels),self._removeEmpty(data,labels)[0],self._removeEmpty(data,labels)[1])
    
  def fit(self, data, labels):
    """Cleans data and computes the bag-of-words vectors.

    Parameters
    ----------
    data: list of tweets where each tweet is a list of the individual word strings

    labels: array, shape: (len(data), )
        Array of the corresponding labels for each tweet.
 
    """
    return(self._clean(data,labels),self._makeSparse(data,labels))

"""### `findBOW`"""

def findBOW(data,labels):
  """must already call BOWPrep and have T defined""" 
  for t in range(0,len(data)):
    T.removeVocab(data[t])
  weights, sparse_mat = T.fit(data,labels)
  weight = weights[0]
  data_clean = weights[1]
  labels_clean = weights[2]
  return(weight,data_clean,labels_clean,sparse_mat)

"""### `RWMD`"""

def asymmRWMD(valTweet,trainTweet,embeddingDictionary):
  dists = []
  mins = []
  wts=[]
  counts_val = collections.Counter(valTweet)
  nv = len(valTweet)
  for x in range(0,nv):
    for w in trainTweet:
      if valTweet[x] == w:
        dists.append(0.0)
        break
      else:
        dists.append(np.linalg.norm(embeddingDictionary[valTweet[x]]-embeddingDictionary[w]))
    mins.append(min(dists))
    wts.append(counts_val[valTweet[x]]/nv)
    dists = []
  return np.dot(np.array(wts), np.array(mins))

def RWMD(valTweet,trainTweet,embeddingDictionary):
  return (asymmRWMD(valTweet, trainTweet, embeddingDictionary)+asymmRWMD(trainTweet, valTweet, embeddingDictionary))/2

"""### `wmd`"""

def wmd(i, row, X_train):

    union_idx = np.union1d(X_train[i].indices, row.indices)
    W_minimal = W_embedding[union_idx]
    W_dist = euclidean_distances(W_minimal)
    bow_i = X_train[i, union_idx].A.ravel()
    bow_j = row[:, union_idx].A.ravel()
    return emd(bow_i, bow_j, W_dist)

"""### Testing Functions"""

def get_query(agg=False):
  if agg:
    i = np.random.choice(len(X2a_test))
    return X2a_test[i], ya_test[i], i
  else:
    i = np.random.choice(len(X2na_test))
    return X2na_test[i], yna_test[i], i

def mybow(twt, v_dict):
  trim_twt = []
  bow = [0]*len(v_dict)
  for w in twt:
    try:
      e = v_dict[w]
      trim_twt.append(w)
    except:
      pass
  if (len(trim_twt)>0):
    n = len(trim_twt)
    counts = collections.Counter(trim_twt)
    for w in trim_twt:
      bow[v_dict[w]] = counts[w]/n
  else:
    bow = None
  return bow

def weighted_word_vec(embedding,weights):
  wwv = {}
  for x in range(0,len(weights)):
    wwv[x] = np.average(embedding,axis=0,weights=weights[x])
  return(wwv)

def compute_wcds(queryTweet, queryLabel, trainWeights):
  xq = T._dataPrep([queryTweet],queryLabel)
  wwv_tr = weighted_word_vec(W_embedding,trainWeights)
  wwv_q = weighted_word_vec(W_embedding,xq)
  w = []
  for t in range(len(wwv_tr)):
    w.append(np.linalg.norm(wwv_q[0]-wwv_tr[t]))
  return np.asarray(w)

def sort(myArr):
  return(np.argsort(myArr),list(np.array(myArr)[np.argsort(myArr)]))

"""must define (vocab_list, vocab_dict, knn, X2, W_embed_dict, T) before calling next function""" 

def find_k_nearest(queryTweet, queryLabel, queryIndex, trainWeights, trainLabels, k, s_train, agg=False):
  qi = queryIndex
  if agg:
    s_xq = sXa_test
  else:
    s_xq = sXna_test
  wcds = compute_wcds(queryTweet, queryLabel, trainWeights)
  idx_sorted, c_dists = sort(wcds)
  found_knn = list(idx_sorted[:k])
  wmdknn = []
  for x in range(0,k):
    wmdknn.append(wmd(qi,s_train[idx_sorted[x]],s_xq))
  initial_order, k_wmds = [], []
  initial_order, k_wmds = sort(wmdknn)
  found_knn = list(np.array(found_knn)[initial_order])
  last_l = bisect_left(c_dists, k_wmds[k-1])
  
  for l in range(k, len(c_dists)):
    m = idx_sorted[l]
    if l < last_l:
      if (RWMD(queryTweet, X2[m], W_embed_dict) < k_wmds[k-1]): 
        tmp_dist = wmd(qi,s_train[m],s_xq)
        if (tmp_dist < k_wmds[k-1]):
          k_wmds.insert(bisect_left(k_wmds,tmp_dist), tmp_dist)
          found_knn.insert(k_wmds.index(tmp_dist), m) 
          k_wmds.pop()
          found_knn.pop()
          last_l = bisect_left(c_dists[:last_l], tmp_dist)
    else:
      break
  return found_knn, k_wmds

def ns_max(dists):
  den = sum(np.exp(-np.array(dists)))
  return (1/den)*np.exp(-np.array(dists))

"""must define (X_train, y_train) or (X_train_s, y_train_s) and variables needed for find_k_nearest before calling next function""" 
  
def test_accuracy(n_predictions, k, small=False, agg=False):
  if small:
    strain = sX_train_s
  else:
    strain = sX_train
  results = []
  if agg:
    for z in range(n_predictions):
      if small:
        result_ixs, result_dists = find_k_nearest(X2a_test[z], ya_test[z], z, wT_s, y_train_s, k, strain, agg=True)
        results.append(ya_test[z] - np.floor(np.dot(ns_max(result_dists), y_train_s[result_ixs])+0.5)==0)
      else:
        result_ixs, result_dists = find_k_nearest(X2a_test[z], ya_test[z], z, wT, y_train, k, strain, agg=True)
        results.append(ya_test[z] - np.floor(np.dot(ns_max(result_dists), y_train[result_ixs])+0.5)==0)
    return sum(results)/n_predictions
  else:
    for z in range(n_predictions):
      if small:
        result_ixs, result_dists = find_k_nearest(X2na_test[z], yna_test[z], z, wT_s, y_train_s, k, strain, agg=False)
        results.append(yna_test[z] - np.floor(np.dot(ns_max(result_dists), y_train_s[result_ixs])+0.5)==0)
      else:
        result_ixs, result_dists = find_k_nearest(X2na_test[z], yna_test[z], z, wT, y_train, k, strain, agg=False)
        results.append(yna_test[z] - np.floor(np.dot(ns_max(result_dists), y_train[result_ixs])+0.5)==0)
    return sum(results)/n_predictions

"""# Load Data"""

start = datetime.now()

"""## [Data Source (Kaggle)](https://www.kaggle.com/dataturks/dataset-for-detection-of-cybertrolls)"""

data_file ='cyberTrolls.json'
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
test_index = test_id = [int(i.strip('\n')) for i in test_file.readlines()]

"""remove test set from full set"""
index = [x for x in all_index if x not in test_index]

test_list = []
for x in range(0,len(test_index)):
  test_list.append(full_content_list[test_index[x]])

content_list = []
for x in range(0,len(index)):
  content_list.append(full_content_list[index[x]])

train_df = data.iloc[index]
X = train_df['content'].tolist()
y = train_df['label'].tolist()
y = [x.decode('utf-8') for x in y]
y = [int(x) for x in y]
y = np.asarray(y,dtype='float64')

test_df = data.iloc[test_index]
X_test = test_df['content'].tolist()
y_test = test_df['label'].tolist()
y_test = [x.decode('utf-8') for x in y_test]
y_test = [int(x) for x in y_test]
y_test = np.asarray(y_test,dtype='float64')

"""### Split Test Into Aggressive and Non-Aggressive"""

agg = test_df.loc[test_df['label'] == b'1']
X_agg = agg['content'].tolist()
y_agg = agg['label'].tolist()
y_agg = [x.decode('utf-8') for x in y_agg]
y_agg = [int(x) for x in y_agg]
y_agg = np.asarray(y_agg,dtype='float64')

n_agg = test_df.loc[test_df['label'] == b'0']
X_nagg = n_agg['content'].tolist()
y_nagg = n_agg['label'].tolist()
y_nagg = [x.decode('utf-8') for x in y_nagg]
y_nagg = [int(x) for x in y_nagg]
y_nagg = np.asarray(y_nagg,dtype='float64')

"""# Upload Word Embedding Info"""

W_df = pd.read_pickle('embedding.pkl')

W_embed_dict = W_df.to_dict('list')
for k in W_embed_dict.keys():
  W_embed_dict[k] = np.asarray(W_embed_dict[k])

W_embedding = list(W_embed_dict.values())
W_embedding = np.array(W_embedding,dtype=np.float64)
vocab_list = list(W_embed_dict.keys())
vocab_dict = {w: k for k, w in enumerate(vocab_list)}

embed_time_file = open('embed_time.txt','r')
embed_time_list = [i for i in embed_time_file]
embed_time = embed_time_list[0]

"""# Computations"""

T = BOWPrep(vocab_list,vocab_dict,W_embed_dict,W_embedding)

dataPrep_s = datetime.now()
wT,X2,y_train,sX_train = findBOW(X,y)
wa_test,X2a_test,ya_test,sXa_test = findBOW(X_agg,y_agg)
wna_test,X2na_test,yna_test,sXna_test = findBOW(X_nagg,y_nagg)
dataPrep_e = datetime.now()
dataPrep_time = dataPrep_e - dataPrep_s

n_pred_a = len(X2a_test)
n_pred_na = len(X2na_test)

acc_s = datetime.now()
na_acc = test_accuracy(n_pred_na,15,small=False,agg=False)
a_acc = test_accuracy(n_pred_a,15,small=False,agg=True)
acc_e = datetime.now()
acc_time = acc_e - acc_s

end = datetime.now()
full_time = end-start

"""# Print Results"""

f = open('tweetSentimentClassification_results.txt','w')
f.write(f'Number of Test Tweets: {len(test_list)} \nEmbedding dimension: {W_embedding.shape[1]} \nVocabulary size: {W_embedding.shape[0]} \nEmbedding time: {embed_time} \nNumber of Predictions: {n_pred_a} \nAccuracy calculations took {acc_time} to compute. \nAccuracy for non aggressive tweets: {na_acc} \nAccuracy for aggressive tweets: {a_acc} \nTotal runtime: {full_time}')
f.close()
