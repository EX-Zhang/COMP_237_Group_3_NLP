
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score

def Dataframe_split(df): # Function to split Dataframe to training and testing set

    df_train = df.sample(frac=0.75)

    df_test = df.drop(df_train.index)

    return df_train['CONTENT'], df_train['CLASS'], df_test['CONTENT'], df_test['CLASS']

# Load Data

Psy_df = pd.read_csv("Youtube01-Psy.csv")

KatyPerry_df = pd.read_csv("Youtube02-KatyPerry.csv")

LMFAO_df = pd.read_csv("Youtube03-LMFAO.csv")

Eminem_df = pd.read_csv("Youtube04-Eminem.csv")

Shakira_df = pd.read_csv("Youtube05-Shakira.csv")

# Data for Model Building

count_vectorizer = CountVectorizer()

Psy_tc = count_vectorizer.fit_transform(Psy_df)

KatyPerry_tc = count_vectorizer.fit_transform(KatyPerry_df)

LMFAO_tc = count_vectorizer.fit_transform(LMFAO_df)

Eminem_tc = count_vectorizer.fit_transform(Eminem_df)

Shakira_tc = count_vectorizer.fit_transform(Shakira_df)

# Downscale the transformed data using tf-idf

tfidf = TfidfTransformer()

Psy_tfidf = tfidf.fit_transform(Psy_tc)

KatyPerry_tfidf = tfidf.fit_transform(KatyPerry_tc)

LMFAO_tfidf = tfidf.fit_transform(LMFAO_tc)

Eminem_tfidf = tfidf.fit_transform(Eminem_tc)

Shakira_tfidf = tfidf.fit_transform(Shakira_tc)

# Shuffle Dataset with frac = 1

Psy_shuffle = Psy_df.sample(frac=1)

KatyPerry_shuffle = KatyPerry_df.sample(frac=1)

LMFAO_shuffle = LMFAO_df.sample(frac=1)

Eminem_shuffle = Eminem_df.sample(frac=1)

Shakira_shuffle = Shakira_df.sample(frac=1)

# Split the dataframe

Psy_train_X, Psy_train_y, Psy_test_X, Psy_test_y = Dataframe_split(Psy_shuffle)

KatyPerry_train_X, KatyPerry_train_y, KatyPerry_test_X, KatyPerry_test_y = Dataframe_split(KatyPerry_shuffle)

LMFAO_train_X, LMFAO_train_y, LMFAO_test_X, LMFAO_test_y = Dataframe_split(LMFAO_shuffle)

Eminem_train_X, Eminem_train_y, Eminem_test_X, Eminem_test_y = Dataframe_split(Eminem_shuffle)

Shakira_train_X, Shakira_train_y, Shakira_test_X, Shakira_test_y = Dataframe_split(Shakira_shuffle)

# Fit the training data into a Naive Bayes classifier

Psy_classifier = GaussianNB()
Psy_classifier.fit(Psy_train_X, Psy_train_y)

KatyPerry_classifier = GaussianNB()
KatyPerry_classifier.fit(KatyPerry_train_X, KatyPerry_train_y)

LMFAO_classifier = GaussianNB()
LMFAO_classifier.fit(LMFAO_train_X, LMFAO_train_y)

Eminem_classifier = GaussianNB()
Eminem_classifier.fit(Eminem_train_X, Eminem_train_y)

Shakira_classifier = GaussianNB()
Shakira_classifier.fit(Shakira_train_X, Shakira_train_y)

# Cross Validation

Psy_cv_accuracy = cross_val_score(Psy_classifier,Psy_shuffle["CONTENT"],Psy_shuffle["CLASS"],scoring='accuracy',cv=5)
print("Psy Accuracy: " + str(round(100*Psy_cv_accuracy.mean(), 2)) + "%")

KatyPerry_cv_accuracy = cross_val_score(KatyPerry_classifier,KatyPerry_shuffle["CONTENT"],KatyPerry_shuffle["CLASS"],scoring='accuracy',cv=5)
print("KatyPerry Accuracy: " + str(round(100*KatyPerry_cv_accuracy.mean(), 2)) + "%")

LMFAO_cv_accuracy = cross_val_score(LMFAO_classifier,LMFAO_shuffle["CONTENT"],LMFAO_shuffle["CLASS"],scoring='accuracy',cv=5)
print("LMFAO Accuracy: " + str(round(100*LMFAO_cv_accuracy.mean(), 2)) + "%")

Eminem_cv_accuracy = cross_val_score(Eminem_classifier,Eminem_shuffle["CONTENT"],Eminem_shuffle["CLASS"],scoring='accuracy',cv=5)
print("Eminem Accuracy: " + str(round(100*Eminem_cv_accuracy.mean(), 2)) + "%")

Shakira_cv_accuracy = cross_val_score(Shakira_classifier,Shakira_shuffle["CONTENT"],Shakira_shuffle["CLASS"],scoring='accuracy',cv=5)
print("Shakira Accuracy: " + str(round(100*Shakira_cv_accuracy.mean(), 2)) + "%")


