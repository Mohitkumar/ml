from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_mldata
import numpy as np


mnist = fetch_mldata('MNIST original', data_home='.')
X = mnist.data
Y = mnist.target

X_train, Y_train, X_test, y_test = X[:60000], Y[:60000], X[60000:], Y[60000:]
shuffle_index = np.random.permutation(60000)
X_train = X_train[shuffle_index]
y_train = Y_train[shuffle_index]

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

print cross_val_score(clf,X_train, y_train, cv=3,scoring='accuracy')

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}]

grid_search = GridSearchCV(clf,param_grid=param_grid,cv=5, scoring='neg_mean_squared_error')

grid_search.fit(X_train, Y_train)
print cross_val_score(grid_search,X_train, y_train, cv=3,scoring='accuracy')

print grid_search.best_estimator_

