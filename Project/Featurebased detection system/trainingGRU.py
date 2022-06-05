import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense

df1 = pd.read_csv('features1.csv')
#print(df.head())
df = df1.drop(['url'],axis=1).copy()

x = df.drop('label',axis=1)
y = df['label']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=1)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train=x_train.values.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.values.reshape(x_test.shape[0],x_test.shape[1],1)

#creating model
model = Sequential()
model.add(GRU(50,return_sequences=True, input_shape=(16,1)))
model.add(GRU(50, return_sequences=True))
model.add(GRU(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=16,verbose=1)

y_pred=model.predict(x_test) 
classes_y=np.round(y_pred).astype(int)

from sklearn.metrics import confusion_matrix
confusion_n = confusion_matrix(y_test,classes_y)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print(accuracy_score(y_test, classes_y))
print(precision_score(y_test, classes_y))
print(recall_score(y_test, classes_y))
print(f1_score(y_test, classes_y))

model.save('modelGRU.h5')