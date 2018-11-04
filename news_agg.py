# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:42:49 2018

@author: Abhishek 
"""

# https://archive.ics.uci.edu/ml/datasets/News+Aggregator
# (b = business, t = science and technology, e = entertainment, m = health) 

import pandas as pd
data = pd.read_csv(r'E:\news-aggregator-dataset\uci-news-aggregator.csv', sep=',',engine='python')

data.columns ='ID TITLE URL PUBLISHER CATEGORY STORY HOSTNAME TIMESTAMP'.split() 
print(data.head())

#data =  data.sample(frac = 0.1)

"""Can we predict the category (business, entertainment, etc.) of a news article given only its headline?
Can we predict the specific story that a news article refers to, given only its headline?"""


data = data.loc[:, ["TITLE", "CATEGORY"]]

import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

"""rev = re.sub('[^a-zA-Z]', ' ', data['TITLE'][0])
rev = rev.lower().split()
stop_words = set(stopwords.words("english"))
rev = [word for word in rev if not word in stop_words]

ps = PorterStemmer()
rev = [ps.stem(word) for word in rev if not word in stop_words]

rev = ' '.join(rev)"""


corpus = []
for i in range(0, data.shape[0]):
    
    TITLE = re.sub('[^a-zA-Z]', ' ', data['TITLE'][i])
    TITLE = TITLE.lower()
    TITLE = TITLE.split()
    ps = PorterStemmer()
    TITLE = [ps.stem(word) for word in TITLE if not word in set(stopwords.words('english'))]
    
    TITLE = ' '.join(TITLE)
    
    corpus.append(TITLE)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(corpus).toarray()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(data["CATEGORY"])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(LogisticRegression())

clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))


import numpy
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import classification_report


kf = KFold(n_splits=5, shuffle=True)

scores = cross_val_score(clf, X, y, cv=kf)

print(scores)
print(numpy.mean(scores))

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=le.classes_))

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print(cm)
print(cr)

import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
# minimize the within-cluster sum of squares (WCSS)
wcss = []
for i in range(1, 6):
    kmeans = MiniBatchKMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 6), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
