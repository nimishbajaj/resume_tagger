#pip install -q -U tensorflow-text

#import tensorflow_text as text
from collections import defaultdict
import glob
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

# Constants
SAMPLE_SIZE = 50000
HEAD = 10
PATH = './resume_corpus'
MAX_LEN = 2000
MAX_WORDS = 5000
EMBEDDING_SIZE = 100
EPOCHS = 50
GLOVE_EMBEDDING = f"./embedding/glove.6B.{EMBEDDING_SIZE}d.txt"


def preprocess_data(path):
    data = defaultdict(dict)
    unique_labels = set()
    i = 0
    for filename in glob.glob(os.path.join(path, '*.txt')):
      name = os.path.basename(filename)
      name = name[:name.index('.')]
      with open(os.path.join(os.getcwd(), filename), mode='r', encoding='windows-1252') as f:
        data[name]["content"] = f.readlines()[0]
        data[name]["id"] = name
      if i==SAMPLE_SIZE:
        break
      i += 1

    i = 0
    for filename in glob.glob(os.path.join(path, '*.lab')):
      name = os.path.basename(filename)
      name = name[:name.index('.')]
      if name not in data:
        continue
      with open(os.path.join(os.getcwd(), filename),  mode='r', encoding='windows-1252') as f:
        labels = list(map(str.strip, f.readlines()))
        if len(labels)!=0:
          data[name]["label"] = labels
          unique_labels.update(labels)
      if i==SAMPLE_SIZE:
        break
      i += 1

    for i, x in enumerate(data):
      print(data[x])
      if i==HEAD:
        break

    df = pd.DataFrame(list(data.values()))
    df = df.dropna()
    for unique_label in unique_labels:
      df[unique_label]=df['label'].apply(lambda x: 1 if unique_label in x else 0)
    df.drop('label', axis=1, inplace=True)

    print(df.columns)
    print("orignal data")
    print(df.head())
    # print(df.describe())
    return df, unique_labels

def clean_data(df, column_name = "content"):
    CLEANR = re.compile('<.*?>')
    def cleanResume(resumeText):
        resumeText = re.sub(CLEANR, '', resumeText)
        resumeText = re.sub('httpS+s*', ' ', resumeText)  # remove URLs
        resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
        resumeText = re.sub('#S+', '', resumeText)  # remove hashtags
        resumeText = re.sub('@S+', '  ', resumeText)  # remove mentions
        resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ',
                            resumeText)  # remove punctuations
        resumeText = re.sub(r'[^x00-x7f]', r' ', resumeText)
        resumeText = re.sub('s+', ' ', resumeText)  # remove extra whitespace
        return resumeText

    # def removeStopWords(resumeText):
    #     tokens =
    #     out = []
    #     for word in tokens:
    #         if word.lower() not in stopwords:
    #             out.append(word)
    #     return " ".join(out)

    df[column_name] = df[column_name].apply(lambda x: cleanResume(x))
    # df[column_name] = df.content.apply(lambda x: removeStopWords(x))
    #
    print("Cleaned data")
    print(df.head())
    return df

df, unique_labels = preprocess_data(PATH)
df = clean_data(df)

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
x = tf.keras.layers.Dense(10, activation='relu')(x)
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

latest = tf.train.latest_checkpoint(checkpoint_dir)

model.load_weights(latest)

predictions = model.predict(np.expand_dims(x_train[43], 0))

print(tokenizer.sequences_to_texts([x_train[43]]))
print(y_train[43])
print(predictions)

