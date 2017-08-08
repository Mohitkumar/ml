from __future__ import print_function
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.externals import joblib


def load_and_predict():
    f = open("out_agg.csv",'a')
    test = pd.read_csv('/home/mohit/comp_data/test.csv')
    test['siteid'].fillna(-999, inplace=True)
    test['browserid'].fillna("None", inplace=True)
    test['devid'].fillna("None", inplace=True)
    test['datetime'] = pd.to_datetime(test['datetime'])
    test['tweekday'] = test['datetime'].dt.weekday
    test['thour'] = test['datetime'].dt.hour
    site_offer_count = test.groupby(['siteid', 'offerid']).size().reset_index()
    site_offer_count.columns = ['siteid', 'offerid', 'site_offer_count']

    site_cat_count = test.groupby(['siteid', 'category']).size().reset_index()
    site_cat_count.columns = ['siteid', 'category', 'site_cat_count']

    site_mcht_count = test.groupby(['siteid', 'merchant']).size().reset_index()
    site_mcht_count.columns = ['siteid', 'merchant', 'site_mcht_count']

    agg_df = [site_offer_count, site_cat_count, site_mcht_count]

    for x in agg_df:
        test = test.merge(x)

    cols = ['siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid']
    for col in cols:
        lbl = LabelEncoder()
        lbl.fit(list(test[col].values))
        test[col] = lbl.transform(list(test[col].values))
    cols_to_use = list(set(test.columns) - set(['ID', 'datetime', 'click']))

    scaler = StandardScaler().fit(test[cols_to_use])

    stest = scaler.transform(test[cols_to_use])

    clf = joblib.load('comp_agg.pkl')
    test_data = stest
    print("started predicting")
    print("ID,click", file=f, end='\n')
    for i,d in zip(test['ID'],test_data):
        print(i,",",clf.predict_proba(d.reshape(1,-1))[0,-1], file=f, end='\n')


if __name__ == '__main__':
    load_and_predict()
