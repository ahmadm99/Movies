import re
import keras
import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras import layers, models, optimizers
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import tensorflow as tf
import json
import pandas, numpy, string
from keras.preprocessing import text, sequence
from keras.callbacks import History
from keras import layers, models, optimizers
from joblib import dump, load

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from sklearn import model_selection, naive_bayes, svm
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from sklearn.linear_model import RidgeClassifierCV
from scipy.sparse import csr_matrix, lil_matrix


# def accuracy(predictions, Test_Y):
#     i=0
#     result = []
#     for row in predictions:
#         if (row == Test_Y.values[i]).all():
#             result.append(1)
#             continue
#         intersect = row & Test_Y.values[i]
#         sum1 = np.sum(intersect)
#         union = row | Test_Y.values[i]
#         sum2 = np.sum(union)
#         accuracy = sum1/sum2
#         result.append(accuracy)
#         i+=1
#     print(result)
#     print(np.sum(result)/len(result))


def accuracy(predictions, Test_Y):
    i=0
    result = []
    for row in predictions:
        if (row == Test_Y.values[i]).all():
            result.append(1)
            continue
        intersect = [a*b for a, b in zip(row, Test_Y.values[i])]
        sum1 = np.count_nonzero(intersect)
        union = [a+b for a, b in zip(row, Test_Y.values[i])]
        sum2 = np.count_nonzero(union)
        accuracy = min(sum1,sum2)/max(sum1,sum2)
        result.append(accuracy)
        i+=1
    # print(result)
    print(np.sum(result)/len(result) * 100)



df = pd.read_csv('test4.csv', index_col=False)
df = pd.read_csv('dataset_new1.csv', index_col=False)
# df = df.drop('genres', 1)
# df.to_csv('test3.csv', index=False)
#
# df = pd.read_csv('test3.csv', index_col=False)
# df.dropna()
# df.to_csv('test4.csv', index=False)

genres_df = df[['Animation','Comedy','Family','Adventure','Fantasy','Romance','Drama','Action','Crime','Thriller','Horror','History','ScienceFiction','Mystery','War','Foreign','Music','Documentary','Western','TVMovie']]

X = df['overview']
# y = [df['Animation'].values, df['Comedy'].values]
# y = genres_df.values
# X, y = make_classification(n_classes=2)
# y = df[['Animation','Comedy','Family','Adventure','Fantasy','Romance','Drama','Action','Crime','Thriller','Horror','History','ScienceFiction','Mystery','War','Foreign','Music','Documentary','Western','TVMovie']]
y= df[['Animation','Drama','Action','Crime','Adventure','Fantasy']]
# y= df[['Animation','Drama','Crime','Adventure','Fantasy']]

# y = df['Comedy']
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(X,y,test_size=0.2, random_state=42)

Tfidf_vect = TfidfVectorizer(max_features=8000)
Tfidf_vect.fit_transform(df['overview'].values.astype('U'))
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# print(Tfidf_vect.vocabulary_)





# # NB Model
#
# Naive = naive_bayes.MultinomialNB()
# Naive.fit(Train_X_Tfidf,Train_Y)
# # predict the labels on validation dataset
# predictions_NB = Naive.predict(Test_X_Tfidf)
# # Use accuracy_score function to get the accuracy
# print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)

#
#
#
# model = RandomForestClassifier(verbose=True, n_jobs=20)
# # train
# model.fit(Train_X_Tfidf, Train_Y)
# # predict
# predictions = model.predict(Test_X_Tfidf)
# # accuracy
# print("Accuracy = ",accuracy_score(Test_Y,predictions) * 100)



# OneVsRest Model
# MultinominalNB (4 genres) --> 45.2% , Actual --> 44.3%
# MultinominalNB (20 genres) --> 7.1%, Actual --> 13%
# LinerSVC (4 genres) --> 41%
# LinearSVC (20 genres) --> 8.2%, Actual --> 15.6%

model = OneVsRestClassifier(LogisticRegression())
model.fit(Train_X_Tfidf, Train_Y)
# predict the labels on validation dataset
predictions = model.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("One Vs Rest Accuracy -> ",accuracy_score(predictions, Test_Y)*100)
# i=0
# for row in Test_Y.values:
#     print(row)
#     print(predictions[i])
#     i+=1
#     print('\n\n')

# accuracy(predictions,Test_Y)

print(f1_score(Test_Y, predictions, average="micro"))


exit(1)

# SVM

# fit the training dataset on the classifier
# SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', verbose=True)
# SVM.fit(Train_X_Tfidf,Train_Y)
# # predict the labels on validation dataset
# predictions_SVM = SVM.predict(Test_X_Tfidf)
# # Use accuracy_score function to get the accuracy
# print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
#
#



# classifier = MLkNN(k=20)
#
#
# x_train = lil_matrix(Train_X_Tfidf).toarray()
# y_train = lil_matrix(Train_Y).toarray()
# x_test = lil_matrix(Test_X_Tfidf).toarray()
# # train
# classifier.fit(x_train, y_train)
# p = classifier.predict(x_test)
#
# print("MLkNN Accuracy -> ",accuracy_score(p, Test_Y)*100)

# Random Forest
# (4 genres) 43.8%
# (20 genres) 8.9%, Actual --> 19.7%        - double estimators --> 21%



# # # Neural Networks
# model = MLPClassifier(verbose=True, max_iter=20)
# model.fit(Train_X_Tfidf, Train_Y)
# predictions = model.predict(Test_X_Tfidf)
# print("MLP Accuracy -> ", accuracy_score(predictions, Test_Y)*100)
# accuracy(predictions, Test_Y)
#





# Neural Networks

trainX = []
testX = []
for item in Train_X_Tfidf:
    trainX.append(item.toarray()[0])
for item in Test_X_Tfidf:
    testX.append(item.toarray()[0])
print(len(trainX))
print(len(Train_Y))

trainX = np.stack(trainX, axis=0)
testX = np.stack(testX, axis=0)
input_layer = layers.Input((8000,))
output_layer1 = layers.Dense(25, activation="relu")(input_layer)
output_layer2 = layers.Dense(25, activation="relu")(output_layer1)
output_layer3 = layers.Dense(6, activation="sigmoid")(output_layer2)
model = models.Model(inputs=input_layer, outputs=output_layer3)
model.compile(loss='binary_crossentropy',optimizer="adam", metrics=["categorical_accuracy"])
history = model.fit(trainX, Train_Y,epochs=30,verbose=2,validation_data=(testX,Test_Y))
predictions = model.predict(testX)
# print("Neural Network Accuracy -> ",accuracy_score(predictions, Test_Y)*100)

j = 0
for row in predictions:
    for i in range(len(row)):
        if row[i] > 0.5:
            row[i] = int(1)
        else:
            row[i] = int(0)
    row = [int(i) for i in row]
    # Test_Y.values[i] = [int(i) for i in Test_Y.values[i]]
    print(row)
    print(Test_Y.values[j])
    j+=1
    print('\n\n')

accuracy(predictions,Test_Y)


#
# plt.figure()
# plt.grid()
# plt.plot(history.history['loss'])
# plt.savefig('loss.png', bbox_inches='tight')



# Train_X_Tfidf = Train_X_Tfidf.reshape(len(Train_X_Tfidf), 1, Train_X_Tfidf.shape[0])
# Train_Y = Train_Y.reshape(len(Train_Y), 1, Train_Y.shape[1])
# Test_X_Tfidf = Test_X_Tfidf.reshape(len(Test_X_Tfidf), 1, Test_X_Tfidf.shape[1])
# Test_Y = Test_Y.reshape(len(Test_X_Tfidf), 1, Train_Y.shape[1])
#
#
# model = Sequential()
# model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(1, 8000)))
# model.add(LSTM(50, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='mse')
# # model.compile(loss='binary_crossentropy', optimizer='adam')
# model.fit(Train_X_Tfidf,Train_Y)
# predictions = model.predict(Test_X_Tfidf)
# print("LSTM Accuracy -> ",accuracy_score(predictions, Test_Y)*100)




# OneVsRest Model

# model = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, weights='uniform'))
# model.fit(Train_X_Tfidf,Train_Y)
# # predict the labels on validation dataset
# predictions_NB = model.predict(Test_X_Tfidf)
# # Use accuracy_score function to get the accuracy
# print("One Vs Rest Accuracy -> ",accuracy_score(predictions_NB, Test_Y)*100)



# new_str = "the true story of molly bloom, an olympic-class skier who ran the world's most exclusive hig stakes poker game and became an fbi target"
# new_str = [entry.lower() for entry in new_str]
#
# row = new_str
# stop_words = set(nltk.corpus.stopwords.words("arabic") + nltk.corpus.stopwords.words("english"))
# tokenizer = nltk.tokenize.WhitespaceTokenizer()
# tokens = tokenizer.tokenize(str(row))
# wnl = nltk.WordNetLemmatizer()
# lemmatizedTokens = [wnl.lemmatize(t) for t in tokens]
# row = [w for w in lemmatizedTokens if w not in stop_words]
# row = ' '.join(row)
#
# replace = r'[/(){}\[\]|@âÂ,;\?\'\"\*…؟–’،!&\+-:؛-]'
# row = re.sub(replace, " ", row)
# words = nltk.word_tokenize(row)
# words = [word for word in words if word.isalpha()]
# row = ' '.join(words)
#
# new_row = Tfidf_vect.transform(row)
#
# new_prediction = Naive.predict(new_row)
# print(new_prediction)




