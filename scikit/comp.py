from __future__ import print_function
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict,train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib

train = pd.read_csv('/home/mohit/comp_data/train.csv')

train['siteid'].fillna(-999, inplace=True)
train['browserid'].fillna("None", inplace=True)
train['devid'].fillna("None", inplace=True)
train['datetime'] = pd.to_datetime(train['datetime'])
train['tweekday'] = train['datetime'].dt.weekday
train['thour'] = train['datetime'].dt.hour
#train['tminute'] = train['datetime'].dt.minute

cols = ['siteid','offerid','category','merchant','countrycode','browserid','devid']

for col in cols:
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values))
    train[col] = lbl.transform(list(train[col].values))

cols_to_use = list(set(train.columns) - set(['ID','datetime','click']))
train = train.sample(frac=1).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(train[cols_to_use], train['click'], test_size=0.2)


def explore_data_hist():
    train.hist(bins=50, figsize=(20,15))
    plt.show()


def explore_data_relation():
    train.groupby('click').hist()
    print(train.corr())
    pd.scatter_matrix(train, alpha=0.3, figsize=(14, 8), diagonal='kde');
    plt.show()


def classify():
    clf = RandomForestClassifier(n_estimators=15)
    clf.fit(X_train, y_train)
    preds = cross_val_predict(clf,X_train, y_train, cv=3)
    print(confusion_matrix(y_train, preds))
    print(roc_auc_score(y_train, preds))
    return clf


def predict(clf):
    print('#####################')
    predicted = clf.predict(X_test)
    print(confusion_matrix(y_true=y_test, y_pred=predicted))
    print(metrics.classification_report(y_test, predicted))
    print(roc_auc_score(y_test,predicted))



if __name__ == '__main__':
    clf = classify()
    #predict(clf)
    joblib.dump(clf, 'comp.pkl')
    #explore_data_hist()
    #explore_data_relation()
