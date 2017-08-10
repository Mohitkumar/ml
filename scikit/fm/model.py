import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer as DV

def get_test(csv_file):
    train = pd.read_csv(csv_file)
    train['siteid'].fillna(0, inplace=True)
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
    return train

def get_data(csv_file):
    train = pd.read_csv(csv_file)
    train['siteid'].fillna(0, inplace=True)
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
    train = train.sample(frac=1).reset_index(drop=True)
    X_train, X_valid, Y_train, Y_valid = train_test_split(train[cols_to_use], train['click'], test_size=0.2, random_state=2017)
    return np.array(X_train), np.array(X_valid), np.array(Y_train), np.array(Y_valid)


def get_data_vect(csv_file):
    train = pd.read_csv(csv_file)
    train['siteid'].fillna(-999, inplace=True)
    train['browserid'].fillna("None", inplace=True)
    train['devid'].fillna("None", inplace=True)
    train['datetime'] = pd.to_datetime(train['datetime'])
    train['tweekday'] = train['datetime'].dt.weekday
    train['thour'] = train['datetime'].dt.hour
    train['tminute'] = train['datetime'].dt.minute
    y_train = train['click']
    num_train = train.drop(
        ['ID', 'datetime', 'click', 'siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid'],
        axis=1)
    cat_train = train.drop(['ID', 'datetime', 'click', 'tweekday', 'thour', 'tminute'], axis=1)
    cat_train['siteid'] = cat_train['siteid'].astype(int)

    x_cat_train = cat_train.to_dict(orient='records')
    vectorizer = DV(sparse=False, dtype=np.int32)
    vec_x_cat_train = vectorizer.fit_transform(x_cat_train)
    train = np.hstack((num_train, vec_x_cat_train))
    X_train, X_test, y_train, y_test = train_test_split(train, y_train, test_size=0.2)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def print_progress(session, epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


if __name__ == '__main__':
    X_train, X_valid, Y_train, Y_valid = get_data('/home/mohit/comp_data/train2.csv')
    Y_train = np.eye(2)[Y_train]
    Y_valid = np.eye(2)[Y_valid]

    n, p = X_train.shape
    k = 5
    X = tf.placeholder('float', shape=[None, p])
    y = tf.placeholder('float', shape=[None, 2])
    y_true_cls = tf.argmax(y, dimension=1)

    w0 = tf.Variable(tf.zeros([2]))
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
    #y_hat = tf.nn.sigmoid(y_hat)
    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')

    l2_norm = (tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_w, tf.pow(W, 2)),
            tf.multiply(lambda_v, tf.pow(V, 2)))))

    error = tf.nn.softmax_cross_entropy_with_logits(logits=y_hat,labels=y)

    y_pred = tf.nn.softmax(y_hat)
    y_pred_cls = tf.argmax(y_hat, dimension=1)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss = tf.add(error, l2_norm)
    cost = tf.reduce_mean(loss)
    eta = tf.constant(0.001)
    optimizer = tf.train.AdagradOptimizer(eta).minimize(cost)
    N_EPOCHS = 100
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
            print(x_data)
            feed_dict_train = {X: x_data,
                               y: y_data}
            feed_dict_validate = {X: X_valid,
                                  y: Y_valid}
            sess.run(optimizer, feed_dict=feed_dict_train)
            val_loss = sess.run(cost, feed_dict=feed_dict_validate)
            print_progress(sess,epoch,feed_dict_train,feed_dict_validate, val_loss)

        print('Loss: ', sess.run(cost, feed_dict={X: x_data, y: y_data}))
        print('accuracy:', sess.run(accuracy, feed_dict={X: x_data, y: y_data}))
        print('Predictions:', sess.run(y_pred_cls, feed_dict={X: x_data}))
        #saver.save(sess, 'fm_comp')