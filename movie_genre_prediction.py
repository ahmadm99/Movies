# -*- coding: utf-8 -*-
"""Movie_Genre_Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/prateekjoshi565/movie_genre_prediction/blob/master/Movie_Genre_Prediction.ipynb

We will start by importing the required libraries.
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# %matplotlib inline
pd.set_option('display.max_colwidth', 300)

"""### Load Data

Now we will import the data uploaded to our Google drive. I will introduce you to a new method to connect the google drive to a colab environment.
"""

# from google.colab import drive
# drive.mount('/content/drive')

"""Let's load the movie metadata first. Please use '\t' as the separator as it is a tab separated file. """

file_path = "movie.metadata.tsv"
meta = pd.read_csv(file_path, sep = '\t', header = None)

meta.head()

"""There are no headers in this dataset. The first column is the **unique movie id**, the third column is the **name of the movie**, and the last column is the **movie genre(s)**. Let's add column names to these columns. We will not use rest of the columns in this analysis."""

# rename columns
meta.columns = ["movie_id",1,"movie_name",3,4,5,6,7,"genre"]

"""Now we will load the **movie plot dataset** into memory. This data comes in a text file with each row consisting a pair of a movie id and a plot of the movie. We will read it line-by-line."""

file_path_2 = "plot_summaries.txt"
plots = []

with open(file_path_2, 'r', encoding="utf-8") as f:
    reader = csv.reader(f, dialect='excel-tab')
    for row in tqdm(reader):
        plots.append(row)

"""We will then split the movie ids and the plots into 2 separate lists. These lists will then be used to form a dataframe."""

movie_id = []
plot = []

for i in tqdm(plots):
    movie_id.append(i[0])
    plot.append(i[1])

movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot})

"""Let's see what we have got."""

movies.head()

"""So, we have movie id and movie plot in this dataframe. Next we will add movie names and movie genres from the movie metadata by merging the latter to the former based on the column *movie_id*.

### Data Exploration and Pre-processing
"""

# change datatype of 'movie_id'
meta['movie_id'] = meta['movie_id'].astype(str)

# merge meta with movies
movies = pd.merge(movies, meta[['movie_id', 'movie_name', 'genre']], on = 'movie_id')

movies.head()

"""Great! We have added both movie names and genres. However, the genres are in dictionary notation. It will easier to work with it if we can somehow convert into python list. Let's try to do it with the first sample."""

movies['genre'][0]

"""We can't access the genres in this sample just by using *.values( )* just like how we do it with a dictionary because it is not actually a dictionary but a string. So, we will have to convert this string into a dictionary. We will take help of the **json library** here."""

type(json.loads(movies['genre'][0]))

"""As you can see, now it is a python dictionary and we ca easily access its genres by using the following code."""

json.loads(movies['genre'][0]).values()

"""We can use this code to extract all the genres from the movies data. After that we will add the extracted genres as lists of genres back to the movies dataframe."""

genres = []

for i in movies['genre']:
  genres.append(list(json.loads(i).values()))
    
movies['genre_new'] = genres

"""Some of the samples might contain no genre tags. Hence, we should take those samples out as they won't help much in model building."""

# remove samples with 0 genre tags
movies_new = movies[~(movies['genre_new'].str.len() == 0)]

movies_new.shape, movies.shape

"""Only 411 samples had no genre tags. Let's check out the dataframe once again. """

movies_new.head()

"""The genres are now in the list format which is going to be helpful since we will have to access these genre tags later.

If you are curious to know which all movie genres have been covered in this dataset, then the following code will be useful for you.
"""

# get all genre tags in a list
all_genres = sum(genres,[])
len(set(all_genres))

"""There are over 363 unique genre tags in our dataset. That is quite a big number. I can hardy recall 5-6 genres. So, let's find out what are these tags. We will use **FreqDist( )** of the nltk library to create a dictionary of genres and their occurence count across the dataset."""

all_genres = nltk.FreqDist(all_genres)
all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()), 'Count': list(all_genres.values())})

"""Sometimes visualizing data is better than putting out numbers. Let's plot the distribution of the movie genres."""

g = all_genres_df.nlargest(columns="Count", n = 50) 
plt.figure(figsize=(12,15))
ax = sns.barplot(data=g, x= "Count", y = "Genre")
ax.set(ylabel = 'Count')
plt.show()

"""As expected, the most frequent tags are "Drama", "Comedy", "Romance", "Thriller", and "Action". The rest of the genres in the data are in some way or other derived from the top 5 genres. 

So, now you can decide whether you want to work with a certain number of most frequent genres or all the genres. As I am solving this problem for the first time, I'd consider all the 363 genres.

Next we will clean our a bit. I will use some very basic text cleaning steps as that is not the focus area of this article.
"""

# function for text cleaning
def clean_text(text):
    # remove backslash-apostrophe
    text = re.sub("\'", "", text)
    # remove everything alphabets
    text = re.sub("[^a-zA-Z]"," ",text)
    # remove whitespaces
    text = ' '.join(text.split())
    # convert text to lowercase
    text = text.lower()
    
    return text

"""Let's apply the function on the movie plots by using the apply-lambda duo."""

movies_new['clean_plot'] = movies_new['plot'].apply(lambda x: clean_text(x))

"""Feel free to check the new vs old movie plots. Given below are a few random samples."""

movies_new[['plot', 'clean_plot']].sample(3)

"""In the clean_plot column the text is in lowercase and there is also no punctuation mark. So, our text cleaning has worked like a charm. This function below can visualize the words with their frequencies, in a set of documents. Let's use it to find out the most frequent words in the movie plots."""

def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()
  
  fdist = nltk.FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
  
  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(12,15))
  ax = sns.barplot(data=d, x= "count", y = "word")
  ax.set(ylabel = 'Word')
  plt.show()

# print 100 most frequent words
freq_words(movies_new['clean_plot'], 100)

"""Most of the terms in the plot above are stopwords. These stopwords carry less important meaning than other keywords in the text. Therefore, we will remove them from the plots' text. You would have to download the list of stopwords from the nltk library."""

nltk.download('stopwords')

"""Now we can remove the stopwords."""



from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)
  
movies_new['clean_plot'] = movies_new['clean_plot'].apply(lambda x: remove_stopwords(x))


"""Let's again check the most frequent words after the stopwords removal."""

freq_words(movies_new['clean_plot'], 100)

"""As it turns out, more interesting words have now emerged, such as "police", "family", "money", "city" and many more.

### Converting Text to Features

I have earlier mentioned in the article that I will treat this multilable classification problem as a Binary Relevance problem. Hence, now I am going to one hot encode the target variable, i.e. *genre_new* by using sklearn's **MultiLabelBinarizer( )**. Since there are 363 unique genre tags, there are going to be 363 new target variables.
"""



from sklearn.preprocessing import MultiLabelBinarizer

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(movies_new['genre_new'])

# transform target variable
y = multilabel_binarizer.transform(movies_new['genre_new'])

"""We have successfully transformed the target variable and now let's turn our focus to extract features from the cleaned version of the movie plots. I have decided to go ahead with TF-IDF features. You are free to use any other feature extraction method such as Bag-of-Words, word2vec, GloVe, or ELMo. 


I recommend you check out these articles to learn more about different ways of creating features from text:



*   [An Intuitive Understanding of Word Embeddings: From Count Vectors to Word2Vec](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/)
*   [A Step-by-Step NLP Guide to Learn ELMo for Extracting Features from Text](https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/)


"""

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)

"""Please note that I have used the 10,000 most frequent words in the data as my features. You can try any other number as well for the parameter max_features. Before creating TF-IDF features, we will split our data into train and validation sets for training our and evaluating its performance, respectively. 80% of the data samples have been kept in the train set and the rest of the data is in the validation set."""

# split dataset into training and validation set
xtrain, xval, ytrain, yval = train_test_split(movies_new['clean_plot'], y, test_size=0.2, random_state=9)

"""Now we can create features for the train and the validation set."""

# create TF-IDF features
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)

"""### Build Your Movie Genre Prediction Model

As we have already discussed earlier, if we are using the Binary Relevance approach to solve a multilable problem, then we will have to one hot encode the target variable and then build model for each and every one hot encoded target variables. Since we have 363 target variables, we will have to fit 363 different models with the same set of predictors (TF-IDF features). 

Training 363 models can take a considerable amount of time on a modest system. Hence, I will go ahead with the Logistic Regression model as it is quick to train and easy on the limited hardware.
"""

from sklearn.linear_model import LogisticRegression

# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier

# Performance metric
from sklearn.metrics import f1_score

"""We will use sklearn's OneVsRestClassifier class to solve this problem as a Binary Relevance or one-vs-all problem."""

lr = LogisticRegression()
clf = OneVsRestClassifier(lr)

"""Finally we are going to fit our model on the train set"""

# fit model on train data
clf.fit(xtrain_tfidf, ytrain)

"""Once our model is trained, we can then predict movie genres for the validation set. Let's do it."""

# make predictions for validation set
y_pred = clf.predict(xval_tfidf)

"""Let's check out a prediction..."""

y_pred[3]

"""It is a binary one dimensional array of length 363. Basically it is the one hot encoded form of the unique genre tags. We will have to find some way to convert it into movie genre tags. Luckily, sklearn is bcak to our rescue once again. We will use **inverse_transform( )** function along with the MultiLabelBinarizer( ) object to convert the predicted arrays into movie genre tags."""

multilabel_binarizer.inverse_transform(y_pred)[3]

"""Wow! That was smooth.

However, to evaluate our model's overall performance, we will have to take into consideration all the predictions and the entire target variable of the validation set.
"""

# evaluate performance
f1_score(yval, y_pred, average="micro")

"""We get a decent F1 score of 0.315. These predictions were made based on a threshold value of 0.5, which means that the probabilities greater than or equal to 0.5 were converted to 1's and the rest to 0's. Let's try to change this threshold value and see if that helps our model."""

# predict probabilities
y_pred_prob = clf.predict_proba(xval_tfidf)

"""Now set a threshold value."""

t = 0.3 # threshold value
y_pred_new = (y_pred_prob >= t).astype(int)

"""I have tried 0.3 as the threshold value. You may try any other value as well. Let's check the F1 score again on these new predictions."""

# evaluate performance
f1_score(yval, y_pred_new, average="micro")

"""That is a big boost to the performance. A better approach to find the right threshold value would be to use a k-fold cross validation setup and try different threshold values.

### Create Inference Function

Our job is still not over. We have trained our model and we have also made predictions for the validation set. However, we also have to take care of the new data or new movie plots that would come in future. Our movie genre prediction system should be able to take a movie plot in raw form as input and give out its genre tags as output.

To achieve this objective, let's build an inference function. It will take in a movie plot text and follow the steps below:



*   Clean the text
*   Remove stopwords from the cleaned text
*   Extract features from the text
*   Make predictions
*   Return the predicted movie genre tags
"""

def infer_tags(q):
    q = clean_text(q)
    q = remove_stopwords(q)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = clf.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)

"""Let's test this inference function on a few samples from the validation set."""

for i in range(5):
    k = xval.sample(1).index[0]
    print("Movie: ", movies_new['movie_name'][k], "\nPredicted genre: ", infer_tags(xval[k])), print("Actual genre: ",movies_new['genre_new'][k], "\n")