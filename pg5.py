from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# load the dataset
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

# vectorize the documents
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(newsgroups_train.data)
X_test = vectorizer.transform(newsgroups_test.data)

# target labels
y_train = newsgroups_train.target
y_test = newsgroups_test.target
# create a Naive Bayes Classifier model
nb_classifier = MultinomialNB()

# train the model on the training data
nb_classifier.fit(X_train, y_train)
# make predictions on the testing data
y_pred = nb_classifier.predict(X_test)

# calculate the accuracy, precision, and recall of the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# print the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
