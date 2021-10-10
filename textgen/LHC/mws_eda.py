# -*- coding: utf-8 -*-
"""MWS_EDA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1P-3CIIH93msrye2AL_TsK4RouOXlBeUb
"""

# import more libraries as need arise 
import numpy as np 
import pandas as pd 
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords 
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import nltk
nltk.download('stopwords')
import pickle

optimizer = keras.optimizers.Adam(lr=0.01)

# read in the training dataset
url = "https://raw.githubusercontent.com/JoshSilverb/mdst_nlp_2021/master/data/train.csv"
df_all = pd.read_csv(url)
options = ['MWS']
df = df_all.loc[df_all['author'].isin(options)]

# fetch author data
def makeauthorfile(options) :
  df_author = df_all.drop(df_all[~df_all['author'].isin(options)].index)
  df_author.drop(columns = ["id", "author"], inplace = True)
  df_author.reset_index(inplace = True)
  df_author.drop(columns = ["index"], inplace = True)
  #print(df_author.head())
  df_author.to_csv('authorfile.txt',index=False)
  authorfile = open('authorfile.txt').read()
  return authorfile

MWSfile = makeauthorfile(options)

#tokenize words for standardization
def tokenize_words(input):
  input = input.lower()

  #instantiate tokenizer
  tokenizer = RegexpTokenizer(r'\w+')
  tokens = tokenizer.tokenize(input)

  filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
  return " ".join(filtered)

processed_inputs = tokenize_words(MWSfile)

chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c, i) for i, c in enumerate(chars))
input_len = len(processed_inputs)
vocab_len = len(chars)
print ("Total number of characters:", input_len)
print ("Total vocab:", vocab_len)

#create dataset
seq_length = 100
x_data =[]
y_data = []

# loop through inputs, start at the beginning and go until we hit
# the final character we can create a sequence out of
for i in range(0, input_len - seq_length, 1):
    # Define input and output sequences
    # Input is the current character plus desired sequence length
    in_seq = processed_inputs[i:i + seq_length]

    # Out sequence is the initial character plus total sequence length
    out_seq = processed_inputs[i + seq_length]

    # We now convert list of characters to integers based on
    # previously and add the values to our lists
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])

n_patterns = len(x_data)
print ("Total Patterns:", n_patterns)

X = np.reshape(x_data, (n_patterns, seq_length, 1))
X = X/float(vocab_len)
y = np_utils.to_categorical(y_data)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#Saved as filepath
filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]

model.fit(X, y, epochs=4, batch_size=256, callbacks=desired_callbacks)

#Chage learning rate mid training
#K.set_value(model.optimizer.learning_rate, 0.001)
#model.fit(X, y, epochs=4, batch_size=256, callbacks=desired_callbacks)

#Loads file name
filename = "model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

num_to_char = dict((i, c) for i, c in enumerate(chars))

start = np.random.randint(0, len(x_data) - 1)
pattern = x_data[start]

# for preview purposes
print("Random Seed:")
print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")

for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab_len)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = num_to_char[index]

    sys.stdout.write(result)

    pattern.append(index)
    pattern = pattern[1:len(pattern)]
