#https://www.tensorflow.org/text/tutorials/text_generation
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time
import pandas as pd
import keras
import matplotlib.pyplot as plt
import nltk
import re
from nltk.corpus import words, wordnet
nltk.download('words')
nltk.download('wordnet')

url = "https://raw.githubusercontent.com/JoshSilverb/mdst_nlp_2021/master/data/train.csv"
df_all = pd.read_csv(url)
options = ['MWS']
df = df_all.loc[df_all['author'].isin(options)]

def addemoji(text) :
  text = text[1:-1]
  text += '😂'
  return text

# fetch author data
def makeauthorfile(options) :
  df_author = df_all.drop(df_all[~df_all['author'].isin(options)].index)
  df_author.drop(columns = ["id", "author"], inplace = True)
  df_author.reset_index(inplace = True)
  df_author.drop(columns = ["index"], inplace = True)
  #df_author.apply(addemoji)
  #print(df_author.head())
  df_author.to_csv('authorfile.txt',index=False)
  authorfile = open('authorfile.txt').read()
  return authorfile

text = makeauthorfile(options)

'''text_file = makeauthorfile(options)
text = text_file.read()
text = text.replace('\n', '')
lines = text.split(".")
lines = list(map(str.strip, lines))

lines.append('\n'.join([line + '😂 \n' for line in lines]))
#text.write('\n'.join([line + '#' for line in lines]))
text = ' '.join(lines)'''

# length of text is the number of characters in it
print(f'Length of text: {len(text)} characters')
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')
for ch in vocab:
  print(ch)
print('test')

ids_from_chars = preprocessing.StringLookup(
    vocabulary=list(vocab), mask_token=None)

chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

for ids in ids_dataset.take(10):
    print(chars_from_ids(ids).numpy().decode('utf-8'))

seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
# Batch size
BATCH_SIZE = 100

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

#dataset_to_numpy = list(dataset.as_numpy_iterator())
#data_np = np.array(dataset_to_numpy)
#X_train=data_np[:,0,:,:]
#Y_train=data_np[:,1,:,:]

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x
model = MyModel(
    # Be sure the vocabulary size matches the `StringLookup` layers.
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)
#model = keras.models.load_model("model20e.H5")
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 50
history = model.fit(dataset,epochs=EPOCHS, callbacks=[checkpoint_callback])
plt.plot(history.history['accuracy'],label="Accuracy")
plt.plot(history.history['loss'],label="Loss")
plt.legend()

class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states
one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

start = time.time()
states = None
next_char = tf.constant(["Once "])
result = [next_char]
for i in range(5):
  for n in range(500):
    next_char, states = one_step_model.generate_one_step(next_char, states=states)
    if (next_char=='😂'):
      break
    result.append(next_char)

'''for n in range(1000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)'''

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)

#Check Percent Words
if not isinstance(result, str):
  result = result[0].numpy().decode('utf-8')
result.replace('; ', ', ')
wordslist = result.split(" ")
pattern = r'[^A-Za-z0-9]+'
numwords = 0
numnotwords = 0
notwords = []
for word in wordslist:
  word = re.sub(pattern, '', word)
  if (word.lower() in words.words()) or wordnet.synsets(word.lower()):
    numwords += 1
  else:
    numnotwords += 1
    notwords.append(word)

print('Percent real words: '+str(100*numwords/(numwords+numnotwords)))
for word in notwords:
  print(word+'.')

tf.saved_model.save(one_step_model, 'one_step')
