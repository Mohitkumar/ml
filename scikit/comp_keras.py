import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score

train = pd.read_csv('/home/mohit/comp_data/train.csv')


train['siteid'].fillna(-999, inplace=True)
train['browserid'].fillna("None", inplace=True)
train['devid'].fillna("None", inplace=True)
train['datetime'] = pd.to_datetime(train['datetime'])
train['tweekday'] = train['datetime'].dt.weekday
train['thour'] = train['datetime'].dt.hour
train['tminute'] = train['datetime'].dt.minute

site_offer_count = train.groupby(['siteid','offerid']).size().reset_index()
site_offer_count.columns = ['siteid','offerid','site_offer_count']

site_cat_count = train.groupby(['siteid','category']).size().reset_index()
site_cat_count.columns = ['siteid','category','site_cat_count']

site_mcht_count = train.groupby(['siteid','merchant']).size().reset_index()
site_mcht_count.columns = ['siteid','merchant','site_mcht_count']

agg_df = [site_offer_count,site_cat_count,site_mcht_count]

for x in agg_df:
    train = train.merge(x)

for c in list(train.select_dtypes(include=['object']).columns):
    if c != 'ID':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values))
        train[c] = lbl.transform(list(train[c].values))

train = train.sample(int(1e6))
print (train.shape)

cols_to_use = [x for x in train.columns if x not in list(['ID','datetime','click'])]
scaler = StandardScaler().fit(train[cols_to_use])

strain = scaler.transform(train[cols_to_use])
X_train, X_valid, Y_train, Y_valid = train_test_split(strain, train.click, test_size = 0.5, random_state=2017)


print (X_train.shape)
print (X_valid.shape)
print (Y_train.shape)
print (Y_valid.shape)


def keras_model(input_dim):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(input_dim,)))  # layer 1
    model.add(Dense(30, activation='relu'))  # layer 2
    model.add(Dense(2, activation='sigmoid'))  # output
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


callback = EarlyStopping(monitor='val_acc', patience=3)
Y_train = to_categorical(Y_train)
Y_valid = to_categorical(Y_valid)
model = keras_model(X_train.shape[1])
model.fit(X_train, Y_train, 1000, 50, callbacks=[callback],validation_data=(X_valid, Y_valid),shuffle=True)

vpreds = model.predict_proba(X_valid)[:,1]
print roc_auc_score(y_true = Y_valid[:,1], y_score=vpreds)

model.save_weights('first_try.h5')