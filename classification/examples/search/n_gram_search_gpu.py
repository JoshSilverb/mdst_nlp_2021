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


    
    
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from keras.wrappers.scikit_learn import KerasClassifier
sequence_length = 100
max_features = 1000000
# Token locations
call = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=1,
    mode='auto', restore_best_weights=True
)

def create_model(neurons,layers,ngram,active, drop,lr):
    # create model
    tf.keras.backend.clear_session()

    Vectorizer = tf.keras.layers.TextVectorization(output_mode= 'tf_idf',ngrams =ngram)
    with tf.device('/device:CPU:0'):
        Vectorizer.adapt(np.hstack((X,df_kaggle['text'])))
    model = tf.keras.Sequential()
    model.add(Vectorizer)
    
    for n in range(layers):
        model.add(tf.keras.layers.Dense(neurons, activation=active))
        model.add(tf.keras.layers.Dropout(drop))

    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(lr),
              metrics=['accuracy'])
    return model



model = KerasClassifier(build_fn=create_model, epochs = 100, validation_split=.2)
# define the grid search parameters
neurons = [32,64,128,256]
ngram = [1,2,3]
layers =  [1,2]
dropout_rate = [0.0,0.1,0.2,0.3,0.4,0.5]
active = ['relu','tanh','sigmoid','linear']
lr = [1e-3,1e-4,1e-5]
batch_size = [32,64,128,256,512]

param_grid = dict(neurons=neurons,ngram = ngram,layers = layers,
                  drop = dropout_rate, active=active,lr = lr,batch_size=batch_size)

#grid = GridSearchCV(estimator=model, scoring = 'neg_log_loss', param_grid=param_grid, n_jobs=1, cv=3, verbose=3)
grid = RandomizedSearchCV(model, param_grid, n_iter=2,scoring = 'neg_log_loss', n_jobs=1, cv=5, verbose=3)


import multiprocessing

def run_grid(X, y):
    grid_result = grid.fit(X,y,callbacks=[call])
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    d=pd.DataFrame(params)
    d['Mean']=means
    d['Std. Dev']=stds
    d.to_csv('test.csv', mode='a', header = False,  index=False )

    
    return (grid_result.best_score_, grid_result.best_params_)

if __name__ == '__main__':
    for n in range(1):
        p = multiprocessing.Process(
            target=run_grid,
            args=(X, y, )
        )
        p.start()
        p.join()

print(tf.config.experimental.get_memory_info('GPU:0'))
