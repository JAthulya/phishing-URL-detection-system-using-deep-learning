import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense



model = tf.keras.models.load_model('modelGRU.h5')

df1 = pd.read_csv('features.csv')
df = df1.drop(['url'],axis=1).copy()

print(df.head())
x = df.drop('label',axis=1)
y = df['label']
x1=x.values.reshape(x.shape[0],x.shape[1],1)


y_pred = model.predict(x1)
classes_y=np.round(y_pred).astype(int)

from sklearn.metrics import confusion_matrix
confusion_n = confusion_matrix(y,classes_y)

import seaborn as sns
ax = sns.heatmap(confusion_n, annot=True, cmap='Blues')

# ax.set_title('Seaborn Confusion Matrix with labels\n\n');
# ax.set_xlabel('\nPredicted Values')
# ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['True','False'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
# import plotter as plt
# plt.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print(accuracy_score(y, classes_y))
print(precision_score(y, classes_y))
print(recall_score(y, classes_y))
print(f1_score(y, classes_y))
