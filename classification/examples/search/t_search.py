import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from keras.wrappers.scikit_learn import KerasClassifier
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import nltk
from nltk.corpus import stopwords


import os            ##  This module is for "operating system" interfaces
import sys           ##  This module is for functionality relevant to the python run time
path_to_datafolder = 'C:/Users/mjdom/source/repos/mdst_nlp_2021/data'
print(os.listdir(path_to_datafolder))


df = pd.read_csv(path_to_datafolder+ '/train.csv')
df_kaggle = pd.read_csv(path_to_datafolder + '/test.csv')
df_kaggle.head()

#np.hstack((X,df_kaggle['text']))[0]

X = df["text"].copy()
#X = df["text"]

authors = df["author"].copy()

# Label data
y = []
for author in authors:
    if author == "EAP":
        y.append([1, 0, 0])
    if author == "HPL":
        y.append([0, 1, 0])
    if author == "MWS":
        y.append([0, 0, 1])

y = np.array(y)

from tensorflow import keras
from tensorflow.keras import layers

import keras

from keras import backend as K

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)



    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)



class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim, mask_zero=True)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from keras.wrappers.scikit_learn import KerasClassifier
sequence_length = 100
max_features = 1000000
# Token locations
Vectorizer_transformer = tf.keras.layers.TextVectorization(max_tokens=max_features,output_sequence_length=sequence_length) 
Vectorizer_transformer.adapt(np.hstack((X,df_kaggle['text'])))
vocab = Vectorizer_transformer.get_vocabulary()
vocab_size = len(vocab)


def create_model(embed_dim=32,num_heads = 2,ff_dim = 32,dropout_rate = 0.2):
    
    # create model
    tf.keras.backend.clear_session()
    K.clear_session()
    #gc.collect()



    maxlen = sequence_length

    ## Build embedding and transformer
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim,dropout_rate)

    ## Connect Keras Layers
    inputs = tf.keras.Input(shape=(1,), dtype=tf.string) 
    vec = Vectorizer_transformer(inputs)
    x = embedding_layer(vec)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(3, activation="softmax")(x)

    transformer = keras.Model(inputs=inputs, outputs=outputs)
    transformer.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(1e-3),
              metrics=['accuracy'])
    return transformer



import time
# fix random seed for reproducibility
#seed = 15
#np.random.seed(seed)
# load dataset

# create model
model = KerasClassifier(build_fn=create_model, batch_size=64, verbose=0)
# define the grid search parameters
embed_dim = [32,64,128,256]
num_heads = [1,2]
ff_dim =  [32,64,128,256]
dropout_rate = [0.0,0.1,0.2,0.3]
epochs = [1,2]

param_grid = dict(embed_dim=embed_dim,num_heads = num_heads,ff_dim = ff_dim,
                  dropout_rate = dropout_rate, epochs=epochs)
#grid = GridSearchCV(estimator=model, scoring = 'neg_log_loss', param_grid=param_grid, n_jobs=1, cv=3, verbose=3)
grid = RandomizedSearchCV(model, param_grid, n_iter=10,scoring = 'neg_log_loss', n_jobs=1, cv=5, verbose=0)


import multiprocessing

def run_grid(X, y):
    grid_result = grid.fit(X,y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    d=pd.DataFrame(params)
    d['Mean']=means
    d['Std. Dev']=stds
    d.to_csv('my_csv_test.csv', mode='a', header = False,  index=False )

    
    return (grid_result.best_score_, grid_result.best_params_)

if __name__ == '__main__':
    for n in range(10):
        p = multiprocessing.Process(
            target=run_grid,
            args=(X, y, )
        )
        p.start()
        p.join()

print(tf.config.experimental.get_memory_info('GPU:0'))
