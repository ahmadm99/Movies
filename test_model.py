import ast
import pickle
import re

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

stop_words = set(stopwords.words('english'))

def clean_text(text):
    # remove backslash-apostrophe
    text = str(text)
    text = re.sub("\'", "", text)
    # remove everything alphabets
    text = re.sub("[^a-zA-Z]", " ", text)
    # remove whitespaces
    text = ' '.join(text.split())
    # convert text to lowercase
    text = text.lower()

    return text


def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)



def infer_tags(q):
    q = clean_text(q)
    q = remove_stopwords(q)
    q_vec = Tfidf_vect.transform([q])
    q_pred = model.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)


df = pd.read_csv('dataset_new1.csv', index_col=False)
df['genres_list'] = df.genres.apply(lambda x: ast.literal_eval(str(x)))

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(df['genres_list'])
y = multilabel_binarizer.transform(df['genres_list'])



model = pickle.load(open("new_model.pkl", 'rb'))
# vocab = pickle.load(open("vocabulary.pkl", 'rb'))
Tfidf_vect = pickle.load(open("vect.pkl", 'rb'))
# Tfidf_vect = TfidfVectorizer(vocabulary=vocab)

print("Predicted genre: ", infer_tags("funny man hilarious"))
