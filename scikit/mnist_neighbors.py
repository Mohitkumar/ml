import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import recall_score, precision_score, confusion_matrix

mnist = fetch_mldata('MNIST original', data_home='.')
X = mnist.data
Y = mnist.target

X_train, Y_train, X_test, y_test = X[:60000], Y[:60000], X[60000:], Y[60000:]
shuffle_index = np.random.permutation(60000)
X_train = X_train[shuffle_index]
y_train = Y_train[shuffle_index]


clf = KNeighborsClassifier(n_jobs=3)
clf.fit(X_train, Y_train)

print cross_val_score(clf,X_train, y_train, cv=3,scoring='accuracy')
print cross_val_score(clf,X_test, y_test, cv=3,scoring='accuracy')


train_pred = cross_val_predict(clf, X_train, Y_train, cv=3)

print confusion_matrix(y_train, train_pred)
print precision_score(y_train, train_pred)
print recall_score(y_train, train_pred)

test_pred = cross_val_predict(clf, X_test, y_test, cv=3)

print confusion_matrix(y_test, test_pred)
print precision_score(y_test, test_pred)
print recall_score(y_test, test_pred)
