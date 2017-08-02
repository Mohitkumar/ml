from __future__ import print_function
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib


def load_and_predict():
    f = open("out.csv",'a')
    test = pd.read_csv('/home/mohit/comp_data/test.csv')
    test['siteid'].fillna(-999, inplace=True)
    test['browserid'].fillna("None", inplace=True)
    test['devid'].fillna("None", inplace=True)
    test['datetime'] = pd.to_datetime(test['datetime'])
    test['tweekday'] = test['datetime'].dt.weekday
    test['thour'] = test['datetime'].dt.hour
    cols = ['siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid']
    for col in cols:
        lbl = LabelEncoder()
        lbl.fit(list(test[col].values))
        test[col] = lbl.transform(list(test[col].values))
    cols_to_use = list(set(test.columns) - set(['ID', 'datetime', 'click']))
    clf = joblib.load('comp.pkl')
    test_data = test[cols_to_use]
    print("ID,click", file=f, end='\n')
    for i,d in zip(test['ID'],test_data.values):
        print(i,",",clf.predict_proba(d.reshape(1,-1))[0,-1], file=f, end='\n')


if __name__ == '__main__':
    load_and_predict()
