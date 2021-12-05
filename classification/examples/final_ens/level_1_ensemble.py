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

import scipy
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.naive_bayes import ComplementNB

from sklearn.naive_bayes import MultinomialNB


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


encoder = tf.keras.layers.TextVectorization()
encoder.adapt(np.hstack((X,df_kaggle['text'])))

max_features = 1000000
Vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_features, output_mode='tf_idf', ngrams=3)
count_vec = tf.keras.layers.TextVectorization(max_tokens=max_features, output_mode='count', sparse=True, ngrams=1)
tfidf_vec = tf.keras.layers.TextVectorization(max_tokens=max_features, output_mode='tf_idf', sparse=True, ngrams=3)

with tf.device('/device:CPU:0'):
    Vectorizer.adapt(np.hstack((X,df_kaggle['text'])))
    tfidf_vec.adapt(np.hstack((X,df_kaggle['text'])))
    count_vec.adapt(np.hstack((X,df_kaggle['text'])))

vocab = encoder.get_vocabulary()
len(vocab)

tdidf = tf.keras.Sequential([
    tfidf_vec])
count = tf.keras.Sequential([                        
    count_vec])
df = pd.DataFrame(columns = ['model', 'average', 'logloss'])

##################################################################################################
class CNN1d(tf.keras.Model):
    def __init__(self, conv1_filters, conv1_size, conv2_filters, conv2_size, encoder):
        super(CNN1d, self).__init__()

        self.encoder = encoder

        vocab = encoder.get_vocabulary()
        
        self.embedding = tf.keras.layers.Embedding(input_dim=len(vocab),output_dim=128,mask_zero=True)
        

        self.conv1 = tf.keras.layers.Conv1D(filters=conv1_filters,
                            kernel_size=conv1_size,
                            padding="same",
                            activation="relu",
                            data_format="channels_last",
                            )
        self.conv2 = tf.keras.layers.Conv1D(filters=conv2_filters,
                            kernel_size=conv2_size,
                            padding="same",
                            activation="relu",
                            data_format="channels_last",
                            )
        self.global_pool = tf.keras.layers.GlobalMaxPool1D(keepdims=False)
        #self.dense1 = tf.keras.layers.Dense(dense1, activation='relu')
        self.dense2 = tf.keras.layers.Dense(3, activation="softmax")

    def call(self, x, training=False):
        emb = self.encoder(x)
        emb = self.embedding(emb)
        conv1 = self.conv1(emb)
        conv2 = self.conv2(emb)
        z = tf.concat([conv1, conv2], axis=2)
        z = self.global_pool(z)
        #z = self.dense1(z)
        z = self.dense2(z)
        return z
##################################################################################################
class TransformerBlock(tf.keras.layers.Layer):
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



class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim, mask_zero=True)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
##################################################################################################

def convert_sparce(sparse_tensor):

    row  = np.array(sparse_tensor.indices[:,0])
    col  = np.array(sparse_tensor.indices[:,1])
    data = np.array(sparse_tensor.values)
    out = scipy.sparse.coo_matrix((data, (row, col)), shape=(sparse_tensor.shape.as_list()))

    return out



call = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=1,
    mode='auto', restore_best_weights=True
)

def create_ngram():
    model_ngram = tf.keras.Sequential()
    model_ngram.add(Vectorizer)
      
    model_ngram.add(tf.keras.layers.Dense(128, activation='sigmoid'))
    model_ngram.add(tf.keras.layers.Dropout(0.5))
      
    model_ngram.add(tf.keras.layers.Dense(3, activation='softmax'))
      
    model_ngram.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(1e-3),
                metrics=['accuracy'])
    return model_ngram

def create_cnn(conv1_filters, conv1_size, conv2_filters, conv2_size):
    model = CNN1d(conv1_filters, conv1_size, conv2_filters, conv2_size, encoder)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(1e-3),
              metrics=['accuracy']
    )
    return model

def create_lstm():
    LSTM = tf.keras.Sequential()
    LSTM.add(encoder)
    LSTM.add(tf.keras.layers.Embedding(input_dim=len(vocab),output_dim=256,mask_zero=True))
      
    LSTM.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,dropout=0.5,return_sequences=True)))
    LSTM.add(tf.keras.layers.GlobalMaxPool1D())

    LSTM.add(tf.keras.layers.Dropout(0.1))
      
    LSTM.add(tf.keras.layers.Dense(3, activation='softmax'))
      
    LSTM.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(1e-3),
                metrics=['accuracy'])
    
    return LSTM
# define the grid search parameters

def create_transformer():
    sequence_length = 100
    max_features = 1000000
    # Token locations
    Vectorizer_transformer = tf.keras.layers.TextVectorization(max_tokens=max_features,output_sequence_length=sequence_length) 
    Vectorizer_transformer.adapt(np.hstack((X,df_kaggle['text'])))
    vocab = Vectorizer_transformer.get_vocabulary()
    vocab_size = len(vocab)


    embed_dim =32  # Embedding size for each token
    num_heads =1  # Number of attention heads
    ff_dim = 128  # Hidden layer size in feed forward network inside transformer
    maxlen = sequence_length
    dropout_rate = 0.2 # Dropout rate of feed forward network 

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

    transformer = keras.Model(inputs=inputs, outputs=outputs) ##Final Model
    
    transformer.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(1e-3),
              metrics=['accuracy'])
    return transformer

def create_hybrid(conv_filters, conv_size, lstm_units):
    model = tf.keras.Sequential([
      encoder,
    tf.keras.layers.Embedding(
        input_dim=len(vocab),
        output_dim=128,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Conv1D(filters=conv_filters,
                            kernel_size=conv_size,
                            padding="same",
                            activation="relu",
                            data_format="channels_last",
                            ),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True)),
    tf.keras.layers.GlobalMaxPool1D(keepdims=False),
    #tf.keras.layers.Dense(dense_units, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation="softmax")
    ])
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics=['accuracy'])
    return model



#grid = GridSearchCV(estimator=model, scoring = 'neg_log_loss', param_grid=param_grid, n_jobs=1, cv=3, verbose=3)

def run_ngram(X,y):

    kf = KFold(n_splits=10)

    for train_index, test_index in kf.split(X):
        for _ in range(1):
            
            if _ == 0:
                df_train_pred = pd.DataFrame([])
                df_test_pred = pd.DataFrame([])
                df_kaggle_pred = pd.DataFrame([])

            else:
                df_train_pred = pd.read_csv('train_pred10_3run.csv', header = None)
                df_test_pred = pd.read_csv('test_pred10_3run.csv', header = None)
                df_kaggle_pred = pd.read_csv('kaggle_pred10_3run.csv', header = None)

                #df_train_pred = pd.DataFrame([])
                #df_test_pred = pd.DataFrame([])


            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y[train_index]
            y_test = y[test_index]


            ngram = create_ngram()

            ngram.fit(X_train, y_train, epochs=100, batch_size=64,validation_data= (X_test, y_test), callbacks=[call])
            
            ngram_pred = ngram.predict(X_train)
            ngram_pred_test = ngram.predict(X_test)


            df_train_pred_ngram = pd.DataFrame(ngram_pred)
            df_test_pred_ngram = pd.DataFrame(ngram_pred_test)
            
            df_train = pd.concat([df_train_pred,df_train_pred_ngram], axis=1)
            df_test = pd.concat([df_test_pred,df_test_pred_ngram], axis=1)


            df_train.to_csv('train_pred10_3run.csv', mode='w', header = False,  index=False )
            df_test.to_csv('test_pred10_3run.csv', mode='w', header = False,  index=False )
            
            kaggle_pred = pd.DataFrame(ngram.predict(df_kaggle['text']))
            df_sub = pd.concat([df_kaggle_pred,kaggle_pred], axis=1)

            df_sub.to_csv('kaggle_pred10_3run.csv', mode='w', header = False,  index=False )

        
        break
def run_ngram_later(X,y):

    kf = KFold(n_splits=10)

    for train_index, test_index in kf.split(X):
        for _ in range(1):
            
            df_kaggle_pred = pd.read_csv('kaggle_pred10_3run.csv', header = None)
            df_train_pred = pd.read_csv('train_pred10_3run.csv', header = None)
            df_test_pred = pd.read_csv('test_pred10_3run.csv', header = None)
            #df_train_pred = pd.DataFrame([])
            #df_test_pred = pd.DataFrame([])

            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y[train_index]
            y_test = y[test_index]


            ngram = create_ngram()

            ngram.fit(X_train, y_train, epochs=100, batch_size=64,validation_data= (X_test, y_test), callbacks=[call])
            
            ngram_pred = ngram.predict(X_train)
            ngram_pred_test = ngram.predict(X_test)


            df_train_pred_ngram = pd.DataFrame(ngram_pred)
            df_test_pred_ngram = pd.DataFrame(ngram_pred_test)
            
            df_train = pd.concat([df_train_pred,df_train_pred_ngram], axis=1)
            df_test = pd.concat([df_test_pred,df_test_pred_ngram], axis=1)


            df_train.to_csv('train_pred10_3run.csv', mode='w', header = False,  index=False )
            df_test.to_csv('test_pred10_3run.csv', mode='w', header = False,  index=False )
            
            kaggle_pred = pd.DataFrame(ngram.predict(df_kaggle['text']))
            df_sub = pd.concat([df_kaggle_pred,kaggle_pred], axis=1)

            df_sub.to_csv('kaggle_pred10_3run.csv', mode='w', header = False,  index=False )
        
        break
def run_cnn(X,y):

    kf = KFold(n_splits=10)

    for train_index, test_index in kf.split(X):
        for _ in range(1):
            
            df_train_pred = pd.read_csv('train_pred10_3run.csv', header = None)
            df_test_pred = pd.read_csv('test_pred10_3run.csv', header = None)
            df_kaggle_pred = pd.read_csv('kaggle_pred10_3run.csv', header = None)



            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y[train_index]
            y_test = y[test_index]


            cnn = create_cnn(64, 128, 32, 32)

            cnn.fit(X_train, y_train, epochs=100, batch_size=256,validation_data= (X_test, y_test), callbacks=[call])
            
            cnn_pred = cnn.predict(X_train)
            cnn_pred_test = cnn.predict(X_test)


            df_train_pred_cnn = pd.DataFrame(cnn_pred)
            df_test_pred_cnn = pd.DataFrame(cnn_pred_test)
            
            df_train = pd.concat([df_train_pred,df_train_pred_cnn], axis=1)
            df_test = pd.concat([df_test_pred,df_test_pred_cnn], axis=1)


            df_train.to_csv('train_pred10_3run.csv', mode='w', header = False,  index=False )
            df_test.to_csv('test_pred10_3run.csv', mode='w', header = False,  index=False )
            
            kaggle_pred = pd.DataFrame(cnn.predict(df_kaggle['text']))
            df_sub = pd.concat([df_kaggle_pred,kaggle_pred], axis=1)

            df_sub.to_csv('kaggle_pred10_3run.csv', mode='w', header = False,  index=False )
        break

def run_lstm(X,y):

    kf = KFold(n_splits=10)

    for train_index, test_index in kf.split(X):
        for _ in range(1):
            
            df_train_pred = pd.read_csv('train_pred10_3run.csv', header = None)
            df_test_pred = pd.read_csv('test_pred10_3run.csv', header = None)
            df_kaggle_pred = pd.read_csv('kaggle_pred10_3run.csv', header = None)



            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y[train_index]
            y_test = y[test_index]


            LSTM = create_lstm()

            LSTM.fit(X_train, y_train, epochs=100, batch_size=64,validation_data= (X_test, y_test), callbacks=[call])
            
            LSTM_pred = LSTM.predict(X_train)
            LSTM_pred_test = LSTM.predict(X_test)


            df_train_pred_LSTM = pd.DataFrame(LSTM_pred)
            df_test_pred_LSTM = pd.DataFrame(LSTM_pred_test)
            
            df_train = pd.concat([df_train_pred,df_train_pred_LSTM], axis=1)
            df_test = pd.concat([df_test_pred,df_test_pred_LSTM], axis=1)


            df_train.to_csv('train_pred10_3run.csv', mode='w', header = False,  index=False )
            df_test.to_csv('test_pred10_3run.csv', mode='w', header = False,  index=False )
            
            kaggle_pred = pd.DataFrame(LSTM.predict(df_kaggle['text']))
            df_sub = pd.concat([df_kaggle_pred,kaggle_pred], axis=1)

            df_sub.to_csv('kaggle_pred10_3run.csv', mode='w', header = False,  index=False )        
        break
        
        
def run_transf(X,y):

    kf = KFold(n_splits=10)

    for train_index, test_index in kf.split(X):
        for _ in range(1):
            
            df_train_pred = pd.read_csv('train_pred10_3run.csv', header = None)
            df_test_pred = pd.read_csv('test_pred10_3run.csv', header = None)
            df_kaggle_pred = pd.read_csv('kaggle_pred10_3run.csv', header = None)



            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y[train_index]
            y_test = y[test_index]


            transformer = create_transformer()

            transformer.fit(X_train, y_train, epochs=100, batch_size=128,validation_data= (X_test, y_test), callbacks=[call])
            
            transformer_pred = transformer.predict(X_train)
            transformer_pred_test = transformer.predict(X_test)


            df_train_pred_transformer = pd.DataFrame(transformer_pred)
            df_test_pred_transformer = pd.DataFrame(transformer_pred_test)
            
            df_train = pd.concat([df_train_pred,df_train_pred_transformer], axis=1)
            df_test = pd.concat([df_test_pred,df_test_pred_transformer], axis=1)


            df_train.to_csv('train_pred10_3run.csv', mode='w', header = False,  index=False )
            df_test.to_csv('test_pred10_3run.csv', mode='w', header = False,  index=False )
            
            kaggle_pred = pd.DataFrame(transformer.predict(df_kaggle['text']))
            df_sub = pd.concat([df_kaggle_pred,kaggle_pred], axis=1)

            df_sub.to_csv('kaggle_pred10_3run.csv', mode='w', header = False,  index=False )    
        break
def run_hybrid(X,y):

    kf = KFold(n_splits=10)

    for train_index, test_index in kf.split(X):
        for _ in range(1):
            
            df_train_pred = pd.read_csv('train_pred10_3run.csv', header = None)
            df_test_pred = pd.read_csv('test_pred10_3run.csv', header = None)
            df_kaggle_pred = pd.read_csv('kaggle_pred10_3run.csv', header = None)



            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y[train_index]
            y_test = y[test_index]


            hybrid = create_hybrid(64, 32, 64)

            hybrid.fit(X_train, y_train, epochs=100, batch_size=32,validation_data= (X_test, y_test), callbacks=[call])
            
            hybrid_pred = hybrid.predict(X_train)
            hybrid_pred_test = hybrid.predict(X_test)


            df_train_pred_hybrid = pd.DataFrame(hybrid_pred)
            df_test_pred_hybrid = pd.DataFrame(hybrid_pred_test)
            
            df_train = pd.concat([df_train_pred,df_train_pred_hybrid], axis=1)
            df_test = pd.concat([df_test_pred,df_test_pred_hybrid], axis=1)


            df_train.to_csv('train_pred10_3run.csv', mode='w', header = False,  index=False )
            df_test.to_csv('test_pred10_3run.csv', mode='w', header = False,  index=False )
            
            kaggle_pred = pd.DataFrame(hybrid.predict(df_kaggle['text']))
            df_sub = pd.concat([df_kaggle_pred,kaggle_pred], axis=1)

            df_sub.to_csv('kaggle_pred10_3run.csv', mode='w', header = False,  index=False )            
        break
        
def run_nbs(X,y):

    kf = KFold(n_splits=10)

    for train_index, test_index in kf.split(X):
        for _ in range(1):
            
            df_train_pred = pd.read_csv('train_pred10_3run.csv', header = None)
            df_test_pred = pd.read_csv('test_pred10_3run.csv', header = None)
            df_kaggle_pred = pd.read_csv('kaggle_pred10_3run.csv', header = None)



            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            
            x_train_sparce = tdidf.predict(X_train)
            x_test_sparce = tdidf.predict(X_test)
            x_train_sparce_count = count.predict(X_train)
            x_test_sparce_count = count.predict(X_test)


            train_count_data = convert_sparce(x_train_sparce_count)
            test_count_data = convert_sparce(x_test_sparce_count)

            
            multi_nb = MultinomialNB(alpha=1.5)
            multi_nb.fit(train_count_data, np.argmax(y_train, axis =1))
            multi_nb_pred = multi_nb.predict_proba(train_count_data)
            multi_nb_pred_test = multi_nb.predict_proba(test_count_data)
            
            
            com_nb = ComplementNB(alpha = 0.9)
            com_nb.fit(train_count_data, np.argmax(y_train, axis =1))
            com_nb_pred = com_nb.predict_proba(train_count_data)
            com_nb_pred_test = com_nb.predict_proba(test_count_data)



            df_train_pred_multi_nb = pd.DataFrame(multi_nb_pred)
            df_test_pred_multi_nb = pd.DataFrame(multi_nb_pred_test)

            df_train_pred_com_nb = pd.DataFrame(com_nb_pred)
            df_test_pred_com_nb = pd.DataFrame(com_nb_pred_test)
            
            df_train = pd.concat([df_train_pred,df_train_pred_multi_nb,df_train_pred_com_nb], axis=1)
            df_test = pd.concat([df_test_pred,df_test_pred_multi_nb,df_test_pred_com_nb], axis=1)


            df_train.to_csv('train_pred10_3run.csv', mode='w', header = False,  index=False )
            df_test.to_csv('test_pred10_3run.csv', mode='w', header = False,  index=False )
            
            
            kaggle_sparce = convert_sparce(tdidf.predict(df_kaggle['text']))
            kaggle_sparce_count = convert_sparce(count.predict(df_kaggle['text']))
            
            multi_nb_kaggle_pred = pd.DataFrame(multi_nb.predict_proba(kaggle_sparce_count))          
            com_nb_kaggle_pred = pd.DataFrame(com_nb.predict_proba(kaggle_sparce_count))
            
            df_sub = pd.concat([df_kaggle_pred,multi_nb_kaggle_pred,com_nb_kaggle_pred], axis=1)
            df_sub.to_csv('kaggle_pred10_3run.csv', mode='w', header = False,  index=False )           
        break
        
        
import multiprocessing
nltk.download('stopwords')
stop = stopwords.words('english')
len(stop)
s_upper = []
for s in stop:
    s_upper.append(s[0].upper()+s[1:])

all_stop = stop + s_upper

X_stop = X.apply(lambda x: ' '.join([word for word in x.split() if word not in (all_stop)]))



if __name__ == '__main__':
    for n in range(3):
        if n == 0:
            
            p = multiprocessing.Process(
                target=run_ngram,
                args=(X, y, )
            )
            p.start()
            p.join()
            
        else:
            
            p = multiprocessing.Process(
                target=run_ngram_later,
                args=(X, y, )
            )
            p.start()
            p.join()
        
        p = multiprocessing.Process(
            target=run_cnn,
            args=(X, y, )
        )
        p.start()
        p.join()
        
        p = multiprocessing.Process(
            target=run_lstm,
            args=(X, y, )
        )
        p.start()
        p.join()
        
        p = multiprocessing.Process(
            target=run_transf,
            args=(X, y, )
        )        
        p.start()
        p.join()   
        
        p = multiprocessing.Process(
            target=run_hybrid,
            args=(X, y, )
        )
        p.start()
        p.join()
        
        p = multiprocessing.Process(
            target=run_nbs,
            args=(X, y, )
        )
        p.start()
        p.join()
        
    '''for n in range(1):


        p = multiprocessing.Process(
            target=run_ngram_later,
            args=(X_stop, y, )
        )
        p.start()
        p.join()

        p = multiprocessing.Process(
            target=run_cnn,
            args=(X_stop, y, )
        )
        p.start()
        p.join()
        
        p = multiprocessing.Process(
            target=run_lstm,
            args=(X_stop, y, )
        )
        p.start()
        p.join()
        
        p = multiprocessing.Process(
            target=run_transf,
            args=(X_stop, y, )
        )        
        p.start()
        p.join()   
        
        p = multiprocessing.Process(
            target=run_hybrid,
            args=(X_stop, y, )
        )
        p.start()
        p.join()
        
        p = multiprocessing.Process(
            target=run_nbs,
            args=(X_stop, y, )
        )
        p.start()
        p.join()'''
        
        
        
        
        
print(tf.config.experimental.get_memory_info('GPU:0'))
