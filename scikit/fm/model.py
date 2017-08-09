import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def get_data(csv_file):
    train = pd.read_csv(csv_file)
    train['siteid'].fillna(-999, inplace=True)
    train['browserid'].fillna("None", inplace=True)
    train['devid'].fillna("None", inplace=True)
    train['datetime'] = pd.to_datetime(train['datetime'])
    train['tweekday'] = train['datetime'].dt.weekday
    train['thour'] = train['datetime'].dt.hour
    train['tminute'] = train['datetime'].dt.minute

    cols = ['siteid', 'merchant', 'offerid', 'category', 'countrycode', 'browserid', 'devid']

    for col in cols:
        lbl = LabelEncoder()
        lbl.fit(list(train[col].values))
        train[col] = lbl.transform(list(train[col].values))
    cols_to_use = list(set(train.columns) - set(['ID', 'datetime', 'click']))

    X_train, X_valid, Y_train, Y_valid = train_test_split(train[cols_to_use], train['click'], test_size=0.5, random_state=2017)
    return np.array(X_train), np.array(X_valid), np.array(Y_train), np.array(Y_valid)

if __name__ == '__main__':
    X_train, X_valid, Y_train, Y_valid = get_data('/home/mohit/comp_data/train2.csv')
    Y_train.shape += (1,)
    Y_valid.shape += (1,)
    n, p = X_train.shape
    k = 5
    X = tf.placeholder('float', shape=[n, p])
    y = tf.placeholder('float', shape=[n, 1])

    w0 = tf.Variable(tf.zeros([1]))
    W = tf.Variable(tf.zeros([p]))
    V = tf.Variable(tf.random_normal([k, p], stddev=0.01))
    linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(W, X), 1, keep_dims=True))
    interactions = (tf.multiply(0.5,
                                tf.reduce_sum(
                                    tf.subtract(
                                        tf.pow(tf.matmul(X, tf.transpose(V)), 2),
                                        tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(V, 2)))),
                                    1, keep_dims=True)))
    y_hat = tf.add(linear_terms, interactions)
    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')

    l2_norm = (tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_w, tf.pow(W, 2)),
            tf.multiply(lambda_v, tf.pow(V, 2)))))

    error = tf.reduce_mean(tf.square(tf.subtract(y, y_hat)))
    loss = tf.add(error, l2_norm)
    eta = tf.constant(0.01)
    optimizer = tf.train.AdagradOptimizer(eta).minimize(loss)
    N_EPOCHS = 100000
    # Launch the graph.
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    msg = "Epoch {0} --- Validation Loss: {3:.3f}"
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(N_EPOCHS):
            indices = np.arange(n)
            np.random.shuffle(indices)
            x_data, y_data = X_train[indices], Y_train[indices]
            sess.run(optimizer, feed_dict={X: x_data, y: y_data})

        print('MSE: ', sess.run(error, feed_dict={X: x_data, y: y_data}))
        print('Loss (regularized error):', sess.run(loss, feed_dict={X: x_data, y: y_data}))
        print('Predictions:', sess.run(y_hat, feed_dict={X: x_data}))
        saver.save(sess, 'fm_comp')