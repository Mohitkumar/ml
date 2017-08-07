from __future__ import print_function
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict,train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer as DV
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.feature_selection import chi2, SelectKBest

train = pd.read_csv('/home/mohit/comp_data/train2.csv')


train['siteid'].fillna(-999, inplace=True)
train['browserid'].fillna("None", inplace=True)
train['devid'].fillna("None", inplace=True)
train['datetime'] = pd.to_datetime(train['datetime'])
train['tweekday'] = train['datetime'].dt.weekday
train['thour'] = train['datetime'].dt.hour
train['tminute'] = train['datetime'].dt.minute

num_train = train.drop(['ID','datetime','click','siteid','offerid','category','merchant','countrycode','browserid','devid'], axis=1)
cat_train = train.drop(['ID','datetime','click','tweekday','thour','tminute'], axis=1)
cat_train['siteid'] = cat_train['siteid'].astype(int)

y_train = train['click']

x_cat_train = cat_train.to_dict(orient='records')
vectorizer = DV(sparse=False)
vec_x_cat_train = vectorizer.fit_transform(x_cat_train)
print(num_train.shape)
print(vec_x_cat_train.shape)
train = np.hstack((num_train,vec_x_cat_train))

np.random.seed(42)
shuffle_index = np.random.permutation(len(y_train))
y_train = y_train[shuffle_index]

train = train[shuffle_index]

print(train[0:3,:])
#train = train.sample(frac=1).reset_index(drop=True)


X_train, X_test, y_train, y_test = train_test_split(train, y_train, test_size=0.2)


def classify():
    clf = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.05, loss='deviance', max_depth=6,
              max_features=None, max_leaf_nodes=None,
              min_impurity_split=1e-07, min_samples_leaf=500,
              min_samples_split=4, min_weight_fraction_leaf=0.0,
              n_estimators=300, presort='auto', random_state=None,
              subsample=1.0, verbose=True, warm_start=False)

    clf.fit(X_train, y_train)
    print(clf.feature_importances_)
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
    #joblib.dump(clf, 'comp_ada.pkl')
