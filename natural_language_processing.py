# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
# quoting = 3 : removes the quotes from the reviews
#print(dataset)
# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
review = review.lower().split()
stop_words = set(stopwords.words("english"))
review = [word for word in review if not word in stop_words]

ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in stop_words]

review = ' '.join(review)


corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #print(review)
    review = ' '.join(review)
    #print(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1200)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Naive Bayes to the Training set
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

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
X_tf = tf.fit_transform(corpus).toarray()
y_tf = dataset.iloc[:, 1].values

tf_X_train, tf_X_test, tf_y_train, tf_y_test = train_test_split(X_tf, y_tf, test_size = 0.25, random_state = 101)

from sklearn.naive_bayes import MultinomialNB
tf_classifier = MultinomialNB()
tf_classifier.fit(tf_X_train, tf_y_train)

# Predicting the Test set results
tf_y_pred = tf_classifier.predict(tf_X_test)

tf_cm = confusion_matrix(tf_y_test, tf_y_pred)
tf_cr = classification_report(tf_y_test, tf_y_pred)
print(tf_cm)
print(tf_cr)

