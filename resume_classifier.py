#pip install -q -U tensorflow-text

#import tensorflow_text as text


import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
wn = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
import re
import utils
from config import *

# Constants


os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(NUM_THREADS)
os.environ["TF_NUM_INTEROP_THREADS"] = str(NUM_THREADS)

tf.config.threading.set_inter_op_parallelism_threads(
    NUM_THREADS
)
tf.config.threading.set_intra_op_parallelism_threads(
    NUM_THREADS
)
tf.config.set_soft_device_placement(True)


df, unique_labels = utils.preprocess_data(PATH)
df = utils.clean_data(df)

def label_freq(df, unique_labels):
    freq = dict()
    for unique_label in unique_labels:
        freq[unique_label] = df[unique_label].mean() * df[unique_label].shape[0]
    print("Label frequency ", freq)
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    plot = sns.barplot(list(freq.keys()), list(freq.values()))
    plot.set_xticklabels(plot.get_xticklabels(),
                         rotation=45,
                         horizontalalignment='right')

tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True)
    tokenizer.fit_on_texts(df['content'])
    sequences = tokenizer.texts_to_sequences(df['content'])
    x = pad_sequences(sequences, maxlen=MAX_LEN)

    train = df
    x_train = train["content"].str.lower()
    y_train = train[unique_labels].values

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_WORDS, lower=True)

    tokenizer.fit_on_texts(x_train)

    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)

    embeddings_index = {}

    with open(GLOVE_EMBEDDING, encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            embed = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embed

    word_index = tokenizer.word_index


    embedding_matrix = np.zeros((MAX_WORDS, EMBEDDING_SIZE), dtype='float32')

    for word, i in word_index.items():

        if i >= MAX_WORDS:
            continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    input = tf.keras.layers.Input(shape=(MAX_LEN,))
    x = tf.keras.layers.Embedding(MAX_WORDS, EMBEDDING_SIZE, weights=[embedding_matrix], trainable=False)(input)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.1,
                                                          recurrent_dropout=0.1))(x)

    x = tf.keras.layers.Conv1D(100, kernel_size=1)(x)

    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)

    x = tf.keras.layers.concatenate([avg_pool, max_pool])
    # x = tf.keras.layers.Dense(10, activation='relu')(x)
    preds = tf.keras.layers.Dense(len(unique_labels), activation="sigmoid")(x)

    model = tf.keras.Model(input, preds)

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])

    batch_size = 128

    checkpoint_path = "./training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        cp_callback
    ]

    model.fit(x_train, y_train, validation_split=0.2, batch_size=batch_size,
              epochs=EPOCHS, callbacks=callbacks, verbose=1)

    model.save(checkpoint_dir, save_format='tf')

    latest = tf.train.latest_checkpoint(checkpoint_dir)

    model.load_weights(latest)

    predictions = model.predict(np.expand_dims(x_train[43], 0))

    print(tokenizer.sequences_to_texts([x_train[43]]))
    print(y_train[43])
    print(predictions)

