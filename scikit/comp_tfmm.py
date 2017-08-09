import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.feature_extraction import DictVectorizer as DV
from tffm import TFFMClassifier
from fm import model

X_train, X_valid, Y_train, Y_valid = model.get_data('/home/mohit/comp_data/train2.csv')

model = TFFMClassifier(
    order=2,
    rank=2,
    optimizer=tf.train.AdadeltaOptimizer(learning_rate=0.01),
    n_epochs=100,
    batch_size=10,
    init_std=0.001,
    input_type='dense')

model.fit(X_train, Y_train, show_progress=True)

pres = model.predict(X_valid)
print roc_auc_score(y_true=Y_valid, y_score=pres)
model.save_state('comp.tf')
model.destroy()