from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.preprocessing import StandardScaler


mnist = fetch_mldata('MNIST original', data_home='.')
X = mnist.data
Y = mnist.target

X_train, Y_train, X_test, y_test = X[:60000], Y[:60000], X[60000:], Y[60000:]
shuffle_index = np.random.permutation(60000)
X_train = X_train[shuffle_index]
y_train = Y_train[shuffle_index]

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

digit = X_test[6000]

pred = clf.predict([digit])
print pred, y_test[6000]

print clf.predict_proba([digit])

print cross_val_score(clf,X_train, y_train, cv=3,scoring='accuracy')

X_train_scaled = StandardScaler().fit_transform(X_train)
clf.fit(X_train_scaled, y_train)
print cross_val_score(clf,X_train, y_train, cv=3,scoring='accuracy')