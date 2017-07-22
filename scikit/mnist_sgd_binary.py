from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original', data_home='.')
X = mnist.data
Y = mnist.target

X_train, Y_train, X_test, Y_test = X[:60000], Y[:60000], X[60000:], Y[60000:]
shuffle_index = np.random.permutation(60000)
X_train = X_train[shuffle_index]
Y_train = Y_train[shuffle_index]

Y_train_5 = (Y_train == 5)
Y_test_5 = (Y_test == 5)

sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(X_train, Y_train_5)

digit = X_test[6000]

pred = sgd_clf.predict([digit])
print pred, Y_test[6000]

print cross_val_score(sgd_clf, X_train, Y_train_5, cv=3, scoring='accuracy')
Y_train_pred = cross_val_predict(sgd_clf, X_train, Y_train_5, cv=3)
print confusion_matrix(y_pred=Y_train_pred, y_true=Y_train_5)
print precision_score(y_pred=Y_train_pred, y_true=Y_train_5)
print recall_score(y_pred=Y_train_pred, y_true=Y_train_5)
print f1_score(y_pred=Y_train_pred, y_true=Y_train_5)

y_scores = cross_val_predict(sgd_clf, X_train, Y_train_5, cv=3,method="decision_function")

precision,recall,threshold = precision_recall_curve(y_true=Y_train_5, probas_pred=y_scores)
plt.plot(threshold, precision[:-1],"b--",label="precision")
plt.plot(threshold, recall[:-1],"g-",label="recall")
plt.xlabel("Threashold")
plt.ylim([0,1])
plt.show()