
import pandas as pd

import nltk, math, string, re

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def comment_preprocessing(comment):

    html_cleaner = re.compile('<a.*?</a>|<br />|<i>.*?</i>|<span class=.*</span>') # regex to match html elements

    comment = re.sub(html_cleaner,'',comment) # replace html elements

    comment = "".join([word for word in comment if word in string.printable]) # remove none ascii characters

    comment = comment.lower()  # to lower case

    comment = "".join([word for word in comment if word not in string.punctuation])  # remove punctuations

    tokens = [word for word in comment.split() if word not in nltk.corpus.stopwords.words("english")]  # remove stopwords

    wnl = nltk.WordNetLemmatizer()

    preprocessed_comment = ' '.join([wnl.lemmatize(token) for token in tokens]) # lemmatize the words

    return preprocessed_comment

# Load Data

LMFAO_df = pd.read_csv("Youtube03-LMFAO.csv")

print("Check if there is missing value:")
print(LMFAO_df.isnull().isnull().sum(),"\n")

df_frames = LMFAO_df[["CONTENT","CLASS"]].sample(frac=1.0)

print("Basic Data Exploration:\n")

print(df_frames.info(),"\n\n",df_frames.describe(),"\n\nShape: ",df_frames.shape,"\n")

df_x = [comment_preprocessing(comment) for comment in df_frames["CONTENT"]]

df_y = df_frames["CLASS"]

index = math.ceil(len(df_x) * 0.75)

df_train_x, df_train_y = df_x[:index], df_y[:index]

df_test_x, df_test_y = df_x[index:], df_y[index:]

# Data for Model Building

count_vectorizer = CountVectorizer()

cv_train_x = count_vectorizer.fit_transform(df_train_x)

cv_test_x = count_vectorizer.transform(df_test_x)

cv_x = count_vectorizer.transform(df_x)

print("Shape of data after vectorized: ", cv_x.shape, "\n")

# Downscale the transformed data using tf-idf

tfidf = TfidfTransformer()

tfidf_train_x = tfidf.fit_transform(cv_train_x)

tfidf_test_x = tfidf.transform(cv_test_x)

tfidf_x = tfidf.transform(cv_x)

print("Shape of data after downscaled using tf-idf: ", tfidf_x.shape, "\n")

# Fit the training data into a Naive Bayes classifier

classifier = MultinomialNB()
classifier.fit(tfidf_train_x,df_train_y)

# Cross Validation

cv_accuracy = cross_val_score(classifier,tfidf_x,df_y,scoring='accuracy',cv=5)
print("Mean Result of Model Accuracy: " + str(round(100*cv_accuracy.mean(), 2)) + "%\n")

# Predict the test data

pred_y = classifier.predict(tfidf_test_x)

print("Confusion Matrix:\n",confusion_matrix(df_test_y,pred_y),"\n")

print("Accuracy of Model: ", classifier.score(tfidf_test_x, df_test_y),"\n")

comments = [ "90 Million views and still going :)",
             "This song is almost 11 years old and it's still incredible",
             "This video's 11th anniversary is today!",
             "I just love David Malan’s teaching style. He’s soooo my favorite professor for CS! How does he make complex topics so easy to understand? Amazing!",
             "Hey, check out my new website!! This site is about kids stuff.kidsmediausa.com",
             "watch?v=vtaRGgvGtWQ Check this out."]

comments_class = [0,0,0,0,1,1]

comments_preprocessed = [comment_preprocessing(comment) for comment in comments]

comments_cv = count_vectorizer.transform(comments_preprocessed)

comments_tfidf = tfidf.transform(comments_cv)

comments_pred = classifier.predict(comments_tfidf)

for i in range(6):

    print(comments[i]," Class: ",comments_class[i]," Predict: ",comments_pred[i],"\n")