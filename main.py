# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# X, y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
# gnb = GaussianNB()
# y_pred = gnb.fit(X_train, y_train).predict(X_test)
# print("Number of mislabeled points out of a total %d points : %d"
#       % (X_test.shape[0], (y_test != y_pred).sum()))

import numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

dataset = pd.read_csv("data/cancer.csv")
dataset.info()
print(dataset.columns)
dataset = dataset.drop(["id"], axis=1)
M = dataset[dataset.diagnosis == "M"]
B = dataset[dataset.diagnosis == "B"]
plt.title("Malignant vs Benign Tumor")
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.scatter(M.radius_mean, M.texture_mean, color="red", label="Malignant", alpha=0.3)
plt.scatter(B.radius_mean, B.texture_mean, color="lime", label="Benign", alpha=0.3)
plt.legend()
plt.show()

dataset.diagnosis = [1 if i == "M" else 0 for i in dataset.diagnosis]
y = dataset.diagnosis.values
x = dataset.drop(["diagnosis"], axis=1)
# Normalization:
x = (x - numpy.min(x)) / (numpy.max(x) - numpy.min(x))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
nb = GaussianNB()
nb.fit(x_train, y_train)
print("Naive Bayes score: ", nb.score(x_test, y_test))
