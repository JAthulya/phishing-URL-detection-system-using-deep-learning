import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['PYTHONHASHSEED'] = '0'

import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
import urllib
from urllib.parse import urlparse, urlencode
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from joblib import dump, load

tf.random.set_seed(42)
np.random.seed(42)

#model = tf.keras.models.load_model('E:\python\phishingurldetectionsystem\Models')
model = tf.keras.models.load_model("modellstm1.h5")

import nltk
import re
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()  

voc_size = 10000
url = 'http://www.slet-fortuna.ru/public_html/file/PHP/chines/95d7e5e8f34c71ff5458a06e29c27d39?login=&_verify%3Fservice=mail&data%3Atext%2Fhtml%3Bcharset=utf-8%3Bbase64%2CPGh0bWw+DQo8c3R5bGU+IGJvZHkgeyBtYXJnaW46IDA7IG92ZXJmbG93OiBoaWRkZW47IH0gPC9zdHlsZT4NCiAgPGlmcmFt'
classes_y=''
status=''
# messages = url
messages = url
f=open("whitelist.txt","r")
if url in f.read():
    classes_y = "this is legitimate"
else:
    status = 1
print(status)
print(classes_y)
if status == 1:
    print('wrong')

print(messages)

corpus=[]

review = re.sub('[^a-zA-Z]',' ',urlparse(messages).netloc)
review = review.lower()
review = review.split()
review=' '.join(review)
corpus.append(review)
print(corpus[0])

# with open('mapping.pkl', 'rb') as fout:
#   mapping = pickle.load(fout)
  
# clf = load('map.joblib')

# mapping = pickle.load(fout)

# onehot_repr = [nltk.word_tokenize(words) for words in corpus]
#corpus, onehot_repr =  zip(*mapping)
#print(mapping)

onehot_repr=[one_hot(words,voc_size)for words in corpus]

#mapping = {c:o for c,o in zip(corpus, onehot_repr)}
print(onehot_repr[0])

sent_length = 50
embedded_docs= pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
#print(embedded_docs[0])

embedded_docs = np.array(embedded_docs)
print(embedded_docs)

x_test = embedded_docs
#y_final  = np.array(y)
#print(x_test)

y_pred = model.predict(x_test)
classes_y=np.round(y_pred).astype(int)

print(y_pred)
print(classes_y)
'''
if classes_y == 0:
    print('not a phishing')
else:
    print('phishing')
    '''
