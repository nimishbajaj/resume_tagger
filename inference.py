import os
import utils
import tensorflow as tf
import numpy as np
from config import *
import shap


checkpoint_dir = os.path.dirname(checkpoint_path)

from tensorflow import keras


def load_model(base_fp=BASE_FP):
    # Model reconstruction from JSON file
    arch_json_fp = '{}-architecture.json'.format(base_fp)
    if not os.path.isfile(arch_json_fp):
        return None

    with open(arch_json_fp, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # Load weights into the new model
    model.load_weights('{}-weights.{}'.format(base_fp, FILE_TYPE))

    print('Loaded model from file ({}).'.format(base_fp))
    return model

model = load_model()
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

background = x_train.iloc[np.random.choice(x_train.shape[0], 100, replace=False)]
explainer = shap.DeepExplainer((model.layers[0].input, model.layers[-1].input), background)

shap_values = explainer.shap_values(x_train[:10].values)

shap.force_plot(explainer.expected_value[0], shap_values[0][0], feature_names=x_train.columns)