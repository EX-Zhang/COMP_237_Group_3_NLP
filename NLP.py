
import pandas as pd

import nltk, math

from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score

# Load Data

Psy_df = pd.read_csv("Youtube01-Psy.csv")

KatyPerry_df = pd.read_csv("Youtube02-KatyPerry.csv")

LMFAO_df = pd.read_csv("Youtube03-LMFAO.csv")

Eminem_df = pd.read_csv("Youtube04-Eminem.csv")

Shakira_df = pd.read_csv("Youtube05-Shakira.csv") 

df_frames = pd.concat([Psy_df, KatyPerry_df, LMFAO_df, Eminem_df, Shakira_df]).sample(frac=1.0)

print("Basic Data Exploration:\n")

print(df_frames.info(),"\n",df_frames.describe(),"\n\nShape: ",df_frames.shape,"\n")

df_x = df_frames["CONTENT"]

df_y = df_frames["CLASS"]

index = math.ceil(len(df_x) * 0.75)

df_train_x, df_train_y = df_x[:index], df_y[:index]

df_test_x, df_test_y = df_x[index:], df_y[index:]

# Data for Model Building

count_vectorizer = CountVectorizer()

cv_train_x = count_vectorizer.fit_transform(df_train_x)

cv_test_x = count_vectorizer.transform(df_test_x)

cv_x = count_vectorizer.transform(df_x)

print("Shape of data after vectorized (Total, Train, Test): ", cv_x.shape, cv_train_x.shape, cv_test_x.shape, "\n")

# Downscale the transformed data using tf-idf

tfidf = TfidfTransformer()

tfidf_train_x = tfidf.fit_transform(cv_train_x)

tfidf_test_x = tfidf.transform(cv_test_x)

tfidf_x = tfidf.transform(cv_x)

print("Shape of data after downscaled using tf-idf (Total, Train, Test): ", tfidf_x.shape, tfidf_train_x.shape, tfidf_test_x.shape, "\n")

# Fit the training data into a Naive Bayes classifier

classifier = MultinomialNB()
classifier.fit(tfidf_train_x,df_train_y)

# Cross Validation

cv_accuracy = cross_val_score(classifier,tfidf_x,df_y,scoring='accuracy',cv=5)
print("Mean Result of Model Accuracy: " + str(round(100*cv_accuracy.mean(), 2)) + "%")

# Predict the test data

pred_y = classifier.predict(tfidf_test_x)

pred_y_flag = pred_y[:] > 0.75

print("Confusion Matrix:\n",confusion_matrix(df_test_y,pred_y_flag))

print("Accuracy of Model: ", classifier.score(tfidf_test_x, df_test_y))
