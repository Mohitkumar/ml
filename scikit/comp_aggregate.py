from __future__ import print_function
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict,train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn import metrics
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib

train = pd.read_csv('/home/mohit/comp_data/train.csv')


train['siteid'].fillna(-999, inplace=True)
train['browserid'].fillna("None", inplace=True)
train['devid'].fillna("None", inplace=True)
train['datetime'] = pd.to_datetime(train['datetime'])
train['tweekday'] = train['datetime'].dt.weekday
train['thour'] = train['datetime'].dt.hour
df1 = train.drop(train[train.click == 0].index)
df2 = train.drop(train[train.click == 1].index)
train = df1.append(df2.iloc[0:len(df1.index)])

cols = ['siteid','merchant','offerid','category','countrycode','browserid','devid']

for col in cols:
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values))
    train[col] = lbl.transform(list(train[col].values))


cols_to_use = list(set(train.columns) - set(['ID','datetime','click']))

train = train.sample(frac=1).reset_index(drop=True)

print(train.groupby('click').count())
X_train, X_test, y_train, y_test = train_test_split(train[cols_to_use], train['click'], test_size=0.2)


def classify():
    clf = GradientBoostingClassifier(criterion='friedman_mse', init=None,
                                     learning_rate=0.05, loss='deviance', max_depth=6,
                                     max_features=None, max_leaf_nodes=None,
                                     min_impurity_split=1e-07, min_samples_leaf=500,
                                     min_samples_split=4, min_weight_fraction_leaf=0.0,
                                     n_estimators=300, presort='auto', random_state=None,
                                     subsample=1.0, verbose=True, warm_start=False)

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
    predict(clf)
    joblib.dump(clf, 'comp_gbc.pkl')
