import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.feature_extraction import DictVectorizer as DV
from tffm import TFFMClassifier

train = pd.read_csv('/home/mohit/comp_data/train2.csv')

train['siteid'].fillna(-999, inplace=True)
train['browserid'].fillna("None", inplace=True)
train['devid'].fillna("None", inplace=True)
train['datetime'] = pd.to_datetime(train['datetime'])
train['tweekday'] = train['datetime'].dt.weekday
train['thour'] = train['datetime'].dt.hour
train['tminute'] = train['datetime'].dt.minute

y_train = train['click']
num_train = train.drop(['ID','datetime','click','siteid','offerid','category','merchant','countrycode','browserid','devid'], axis=1)
cat_train = train.drop(['ID','datetime','click','tweekday','thour','tminute'], axis=1)
cat_train['siteid'] = cat_train['siteid'].astype(int)

x_cat_train = cat_train.to_dict(orient='records')
vectorizer = DV(sparse=False)
vec_x_cat_train = vectorizer.fit_transform(x_cat_train)
train = np.hstack((num_train,vec_x_cat_train))

X_train, X_valid, Y_train, Y_valid = train_test_split(train, y_train, test_size=0.5, random_state=2017)



def keras_model():
    model = TFFMClassifier(
        order=6,
        rank=2,
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
        n_epochs=100,
        batch_size=-1,
        init_std=1.0,
        input_type='dense')
    return model

model = keras_model()

model.fit(X_valid, Y_valid, show_progress=True)

#vpreds = model.predict(X_valid)
#print roc_auc_score(y_true = Y_valid[:,1], y_score=vpreds)

model.destroy()