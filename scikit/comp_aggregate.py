from __future__ import print_function
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

site_offer_count = train.groupby(['siteid','offerid']).size().reset_index()
site_offer_count.columns = ['siteid','offerid','site_offer_count']

site_cat_count = train.groupby(['siteid','category']).size().reset_index()
site_cat_count.columns = ['siteid','category','site_cat_count']

site_mcht_count = train.groupby(['siteid','merchant']).size().reset_index()
site_mcht_count.columns = ['siteid','merchant','site_mcht_count']

agg_df = [site_offer_count,site_cat_count,site_mcht_count]

for x in agg_df:
    train = train.merge(x)

cols = ['siteid','merchant','offerid','category','countrycode','browserid','devid']

for col in cols:
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values))
    train[col] = lbl.transform(list(train[col].values))


cols_to_use = list(set(train.columns) - set(['ID','datetime','click']))

scaler = StandardScaler().fit(train[cols_to_use])

strain = scaler.transform(train[cols_to_use])

#strain = strain.sample(frac=1).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(strain, train['click'], test_size=0.2)


def classify():
    clf = GradientBoostingClassifier(
              learning_rate=0.03,max_features=3,
              n_estimators=300,verbose=True)

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
    joblib.dump(clf, 'comp_agg.pkl')
