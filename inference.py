import os
import utils
import tensorflow as tf
import numpy as np
from config import *

MAX_LEN = 2000
checkpoint_path = "./training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

from tensorflow import keras
model = keras.models.load_model(checkpoint_dir)

df, unique_labels = utils.preprocess_data(PATH)
df = utils.clean_data(df)
train = df
x_train = train["content"].str.lower()
y_train = train[unique_labels].values

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_WORDS, lower=True)

tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_sequences(x_train)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)

predictions = model.predict(np.expand_dims(x_train[43], 0))

print(tokenizer.sequences_to_texts([x_train[43]]))
print(y_train[43])
print(predictions)
