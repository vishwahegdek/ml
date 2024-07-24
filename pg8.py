import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the KNN classifier
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# Train the KNN classifier
knn.fit(X_train, y_train)

# Make predictions on the testing set
predictions = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Print correct and wrong predictions
correct_predictions = []
wrong_predictions = []
for i in range(len(predictions)):
    if predictions[i] == y_test[i]:
        correct_predictions.append((X_test[i], y_test[i], predictions[i]))
    else:
        wrong_predictions.append((X_test[i], y_test[i], predictions[i]))

print("\nCorrect Predictions:")
for prediction in correct_predictions:
    print("Input:", prediction[0], "Actual Class:", iris.target_names[prediction[1]], "Predicted Class:", iris.target_names[prediction[2]])

print("\nWrong Predictions:")
for prediction in wrong_predictions:
    print("Input:", prediction[0], "Actual Class:", iris.target_names[prediction[1]], "Predicted Class:", iris.target_names[prediction[2]])

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(conf_matrix)
