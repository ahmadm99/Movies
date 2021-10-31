import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
# nltk.download('stopwords')

# def clean_text(text):
#     text = re.sub("\'", "", text)
#     text = re.sub("[^a-zA-Z]", " ", text)
#     text = ' '.join(text.split())
#     text = text.lower()
#     return text
#
#
# def remove_stopwords(text):
#     no_stopword_text = [w for w in text.split() if not w in stop_words]
#     return ' '.join(no_stopword_text)
#
#
# df = pd.read_csv('dataset.csv')
# df.overview = df.overview.astype(str)
#
# df['overview'] = df['overview'].apply(lambda x: clean_text(x))
#
# stop_words = set(stopwords.words('english'))
#
# df['overview'] = df['overview'].apply(lambda x: remove_stopwords(x))


# df.to_csv("dataset_new.csv", index=False)









