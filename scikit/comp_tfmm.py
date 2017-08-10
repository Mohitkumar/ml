import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tffm import TFFMClassifier
import pandas as pd

def train():
    from fm import model
    X_train, X_valid, Y_train, Y_valid = model.get_data('/home/mohit/comp_data/train.csv')
    model = TFFMClassifier(
        order=4,
        rank=2,
        optimizer=tf.train.AdadeltaOptimizer(learning_rate=0.01),
        n_epochs=100,
        batch_size=1000,
        init_std=0.0000001,
        input_type='dense')

    model.fit(X_train, Y_train, show_progress=True)

    pres = model.predict(X_valid)
    print roc_auc_score(y_true=Y_valid, y_score=pres)
    model.save_state('fm/comp.tf')
    model.destroy()

def predict_test():
    from fm import model
    test = model.get_test('/home/mohit/comp_data/test.csv')
    model = TFFMClassifier(
        order=4,
        rank=2,
        optimizer=tf.train.AdadeltaOptimizer(learning_rate=0.01),
        n_epochs=1000,
        batch_size=1000,
        init_std=0.0000001,
        input_type='dense')
    model.core.set_num_features(10)
    model.load_state('fm/comp.tf')
    print "start prediciting"
    cols_to_use = list(set(test.columns) - set(['ID', 'datetime', 'click']))
    preds = model.predict_proba(test[cols_to_use])
    print preds.shape
    print preds[0:3,]
    submit = pd.DataFrame({'ID': test.ID, 'click': preds[:,-1]})
    submit.to_csv('fm/pred.csv', index=False)

if __name__ == '__main__':
    train()