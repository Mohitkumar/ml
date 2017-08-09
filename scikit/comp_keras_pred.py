import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler, LabelEncoder


def keras_model(input_dim):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(input_dim,)))  # layer 1
    model.add(Dense(30, activation='relu'))
    model.add(Dense(15, activation='relu'))  # layer 2
    model.add(Dense(2, activation='sigmoid'))  # output
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


test = pd.read_csv('/home/mohit/comp_data/test.csv')


test['siteid'].fillna(-999, inplace=True)
test['browserid'].fillna("None", inplace=True)
test['devid'].fillna("None", inplace=True)
test['datetime'] = pd.to_datetime(test['datetime'])
test['tweekday'] = test['datetime'].dt.weekday
test['thour'] = test['datetime'].dt.hour
test['tminute'] = test['datetime'].dt.minute

site_offer_count = test.groupby(['siteid','offerid']).size().reset_index()
site_offer_count.columns = ['siteid','offerid','site_offer_count']

site_cat_count = test.groupby(['siteid','category']).size().reset_index()
site_cat_count.columns = ['siteid','category','site_cat_count']

site_mcht_count = test.groupby(['siteid','merchant']).size().reset_index()
site_mcht_count.columns = ['siteid','merchant','site_mcht_count']

agg_df = [site_offer_count,site_cat_count,site_mcht_count]

for x in agg_df:
    test = test.merge(x)

for c in list(test.select_dtypes(include=['object']).columns):
    if c != 'ID':
        lbl = LabelEncoder()
        lbl.fit(list(test[c].values))
        test[c] = lbl.transform(list(test[c].values))

print (test.shape)

cols_to_use = [x for x in test.columns if x not in list(['ID','datetime','click'])]
scaler = StandardScaler().fit(test[cols_to_use])

stest = scaler.transform(test[cols_to_use])

model =keras_model(stest.shape[1])
model.load_weights('first_try.h5')
print "started predicting"
test_preds = model.predict_proba(stest)[:,1]
submit = pd.DataFrame({'ID':test.ID, 'click':test_preds})
submit.to_csv('keras_starter.csv', index=False)