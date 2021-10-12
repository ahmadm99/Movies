import re

import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

nltk.download('averaged_perceptron_tagger')

if __name__ == '__main__':
    df = pd.read_csv('dataset.csv')
    df.overview = df.overview.astype(str)
    df['overview'].dropna(inplace=True)
    df['overview'] = [entry.lower() for entry in df['overview']]
    # df.overview.replace(r'[^a-zA-Z]', '', regex=True, inplace=True)
    for i in range(df[df.columns[0]].count()):
        print(i)
        row = df['overview'][i]
        stop_words = set(nltk.corpus.stopwords.words("arabic") + nltk.corpus.stopwords.words("english"))
        tokenizer = nltk.tokenize.WhitespaceTokenizer()
        tokens = tokenizer.tokenize(row)
        wnl = nltk.WordNetLemmatizer()
        lemmatizedTokens = [wnl.lemmatize(t) for t in tokens]
        row = [w for w in lemmatizedTokens if w not in stop_words]
        row = ' '.join(row)

        replace = r'[/(){}\[\]|@âÂ,;\?\'\"\*…؟–’،!&\+-:؛-]'
        row = re.sub(replace, " ", row)
        words = nltk.word_tokenize(row)
        words = [word for word in words if word.isalpha()]
        row = ' '.join(words)

        df.at[i, 'overview'] = row



    df['overview'] = [word_tokenize(entry) for entry in df['overview']]
    df.to_csv('clean.csv')
