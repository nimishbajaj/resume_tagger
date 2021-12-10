from collections import defaultdict
import re
import glob
import os
import pandas as pd
from config import *


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
      # print(data[x])
      if i==HEAD:
        break

    df = pd.DataFrame(list(data.values()))
    df = df.dropna()
    for unique_label in unique_labels:
      df[unique_label]=df['label'].apply(lambda x: 1 if unique_label in x[0] else 0)
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