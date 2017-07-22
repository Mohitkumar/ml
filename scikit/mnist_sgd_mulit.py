from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.model_selection import  cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original', data_home='.')
X = mnist.data
Y = mnist.target

X_train, Y_train, X_test, y_test = X[:60000], Y[:60000], X[60000:], Y[60000:]
shuffle_index = np.random.permutation(60000)
X_train = X_train[shuffle_index]
y_train = Y_train[shuffle_index]

clf = SGDClassifier(random_state=42, loss='log')
clf.fit(X_train, y_train)

digit = X_test[6000]

pred = clf.predict([digit])
print pred, y_test[6000]

scores = clf.decision_function([digit])
print scores
print np.argmax(scores)

print clf.predict_proba([digit])

print cross_val_score(clf,X_train, y_train, cv=3,scoring='accuracy')
preds = cross_val_predict(clf,X_train, y_train, cv=3)

conf_mx = confusion_matrix(y_train, preds)
print conf_mx

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()