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

one_step_reloaded = tf.saved_model.load('./one_step')
os.system('cls' if os.name == 'nt' else 'clear')
starting = input("Enter starting string:")
start = time.time()
states = None
next_char = tf.constant([starting])
result = [next_char]
for i in range(5):
  for n in range(500):
    next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
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
