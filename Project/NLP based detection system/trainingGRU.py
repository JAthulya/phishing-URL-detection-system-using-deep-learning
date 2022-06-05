import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import pandas as pd
import pickle
from urllib.parse import urlparse, urlencode
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense
from numpy.random import seed 
seed(42)
from tensorflow.random import set_seed
set_seed(42)
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
from joblib import dump,load 
# ###set the random seed
# tf.random.set_seed(42)
# np.random.seed(42)

###read and process the data

df = pd.read_csv('finaldata.csv')
x= df['url']
y = df['label']

# df = pd.read_csv('Dataset.csv')
# x= df['url']
# y = df['result']

voc_size = 10000

onehot_dict = {}
messages = x.copy()

ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]',' ',urlparse(messages[i]).netloc)
    #review = re.sub('[^a-zA-Z]',' ',messages[i])
    review = review.lower()
    review = review.split()
    review=' '.join(review)
    corpus.append(review)

onehot_repr=[one_hot(words,voc_size)for words in corpus]

# for words in corpus:
#     onehot_dict[words] = one_hot(words,voc_size)
    
# onehot_df = pd.DataFrame.from_dict(onehot_dict, orient='index')
# onehot_df.to_csv('./onehot.csv',index=False)

#mapping = {c:o for c,o in zip(corpus, onehot_repr)}
# with open('mapping.pkl', 'wb') as fout:
#   pickle.dump(one_hot, fout)

# dump(one_hot, 'map.joblib')

sent_length = 50
embedded_docs= pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

embedded_docs = np.array(embedded_docs)
#x_final = np.array(embedded_docs)
x_final = embedded_docs
y_final  = np.array(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_final,y_final,test_size=0.20)


#make the model and train it
embedding_vector_features=10
model = Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.compile('adam','mse')
model.add(GRU(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=5,batch_size=64)


y_pred=model.predict(x_test) 
classes_y=np.round(y_pred).astype(int)
#model.save('E:\python\phishingurldetectionsystem\Models')
model.save("modelGRU.h5")


# with open('model_pkl', 'wb') as files:
#     pickle.dump((model), files)
    
from sklearn.metrics import confusion_matrix
confusion_n = confusion_matrix(y_test,classes_y)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print(accuracy_score(y_test, classes_y))
print(precision_score(y_test, classes_y))
print(recall_score(y_test, classes_y))
print(f1_score(y_test, classes_y))