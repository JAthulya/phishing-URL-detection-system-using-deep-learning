import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import tensorflow as tf
import numpy as np
from urllib.parse import urlparse, urlencode
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense


model = tf.keras.models.load_model('modelGRU.h5')


df = pd.read_csv('finaldata.csv')
x = df['url']
y = df['label']

# df = pd.read_csv('Dataset.csv')
# x= df['url']
# y = df['result']

voc_size = 10000
messages = x.copy()

import nltk
import re
from nltk.corpus import stopwords



from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]',' ',urlparse(messages[i]).netloc)
    #review = re.sub('[^a-zA-Z]',' ',messages[i])
    review = review.lower()
    review = review.split()
    review=' '.join(review)
    corpus.append(review)
    
#print(corpus[0])
onehot_repr=[one_hot(words,voc_size)for words in corpus]
#print(onehot_repr[1])
sent_length = 50
embedded_docs= pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
#print(embedded_docs[1])


x_final = np.array(embedded_docs)
y_final  = np.array(y)

y_pred = model.predict(x_final)
classes_y=np.round(y_pred).astype(int)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
confusion_n = confusion_matrix(y_final,classes_y)

import seaborn as sns
ax = sns.heatmap(confusion_n, annot=True, cmap='Blues')

#ax.set_title('Seaborn Confusion Matrix with labels\n\n');
#ax.set_xlabel('\nPredicted Values')
#ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1'])
ax.yaxis.set_ticklabels(['0','1'])

## Display the visualization of the Confusion Matrix.
# import plotter as plt
# plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print(accuracy_score(y_final, classes_y))
print(precision_score(y_final, classes_y))
print(recall_score(y_final, classes_y))
print(f1_score(y_final, classes_y))















