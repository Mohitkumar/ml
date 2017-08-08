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
from sklearn.externals import joblib
from sklearn.feature_selection import chi2, SelectKBest

train = pd.read_csv('/home/mohit/comp_data/train.csv')


train['siteid'].fillna(0, inplace=True)
train['browserid'].fillna("None", inplace=True)
train['devid'].fillna("None", inplace=True)
train['datetime'] = pd.to_datetime(train['datetime'])
train['tweekday'] = train['datetime'].dt.weekday
train['thour'] = train['datetime'].dt.hour
train['tminute'] = train['datetime'].dt.minute
df1 = train.drop(train[train['click'] == 0].index)
df2 = train.drop(train[train['click'] == 1].index)
train = df1.append(df2.iloc[0:len(df1.index)])

y_train = train['click']
num_train = train.drop(['ID','datetime','click','siteid','offerid','category','merchant','countrycode','browserid','devid'], axis=1)
cat_train = train.drop(['ID','datetime','click','tweekday','thour','tminute'], axis=1)
cat_train['siteid'] = cat_train['siteid'].astype(int)

x_cat_train = cat_train.to_dict(orient='records')
vectorizer = DV(sparse=False)
vec_x_cat_train = vectorizer.fit_transform(x_cat_train)
train = np.hstack((num_train,vec_x_cat_train))

#np.random.seed(42)
#shuffle_index = np.random.permutation(len(train))
#y_train = y_train[shuffle_index]
#print(np.isnan(y_train).any())
#train = train[shuffle_index]

#train = train.sample(frac=1).reset_index(drop=True)


X_train, X_test, y_train, y_test = train_test_split(train, y_train, test_size=0.2)

def classify():
    clf = GradientBoostingClassifier(
              learning_rate=0.03,max_features=3,
              n_estimators=300,verbose=True)

    clf = Pipeline([('featue_select', SelectKBest(chi2, k=24)), ('classification', clf)])

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
    #joblib.dump(clf, 'comp_gb_hash.pkl')
