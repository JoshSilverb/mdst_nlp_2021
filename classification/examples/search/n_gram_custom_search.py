import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict,KFold
import random
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



# define the grid search parameters
neurons_ = [8,16,24,32,48,64,128,256]
ngram_ = [1,2,3]
layers_ =  [1,2]
dropout_rate_ = [0.0,0.1,0.2,0.3,0.4,0.5]
active_ = ['sigmoid']
lr_ = [1e-3]
batch_size_ = [64,128,256,512]



#grid = GridSearchCV(estimator=model, scoring = 'neg_log_loss', param_grid=param_grid, n_jobs=1, cv=3, verbose=3)

def run_search(X,y):
    df = pd.DataFrame(columns = ['neurons', 'drop', 'ngram', 'layers','batch', 'active','lr', 'mean_loss', 'std_loss', 'mean_acc', 'std_acc'])
    for _ in range(1):
        kf = KFold(n_splits=5)
        results_temp = []
        neurons = random.choice(neurons_)
        ngram = random.choice(ngram_)
        active = random.choice(active_)
        lr = random.choice(lr_)
        drop = random.choice(dropout_rate_)
        batch = random.choice(batch_size_)
        layers = random.choice([1,2])
        for train_index, test_index in kf.split(X):

            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y[train_index]
            y_test = y[test_index]



            m = create_model(neurons,layers,ngram,active, drop,lr)

            m.fit(X_train, y_train, epochs=100, batch_size=batch,validation_data= (X_test, y_test), callbacks=[call],verbose = 0)
            r = m.evaluate(X_test, y_test)
            if r[0]>.39:
                break

            results_temp.append(r)
            del m
            tf.keras.backend.clear_session()


        df_results = pd.DataFrame({"neurons":[neurons],\
                      'drop': [drop]
                      ,'ngram':[ngram]
                      , 'layers':[layers]
                      , 'batch':[batch]
                      , 'active' : [active]
                      , 'lr':[lr]
                      , 'mean_loss':[np.mean(results_temp,axis = 0)[0]]
                      , 'std_loss':[np.std(results_temp,axis = 0)[0]]
                      , 'mean_acc':[np.mean(results_temp,axis = 0)[1]]
                      , 'std_acc':[np.std(results_temp,axis = 0)[1]]})
        df = df.append(df_results)
    df.to_csv('test.csv', mode='a', header = False,  index=False )




import multiprocessing



if __name__ == '__main__':
    for n in range(50):
        p = multiprocessing.Process(
            target=run_search,
            args=(X, y, )
        )
        p.start()
        p.join()

print(tf.config.experimental.get_memory_info('GPU:0'))
