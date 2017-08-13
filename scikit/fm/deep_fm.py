import numpy as np

import tensorflow as tf
from sklearn.metrics import roc_auc_score,confusion_matrix
from model import get_dummy, get_data_vect, get_data_out, get_data,get_data_vect_out
import pandas as pd

def make_embeddings(x, rank, num_features, depth=1, seed=12345):
    """
      assumes that all hidden layers are width `rank`
    """
    assert depth > 0
    V = tf.Variable(tf.truncated_normal([rank, num_features], stddev=0.2, mean=0, seed=seed), name="v_1")
    #10x5
    b = tf.Variable(tf.truncated_normal([rank, 1], stddev=0.2, mean=0, seed=seed), name="b_1")
    #10x1
    Vx = tf.nn.relu(tf.matmul(V, x) + b)
    #10x999
    for i in range(depth - 1):
        V = tf.Variable(tf.truncated_normal([rank, rank], stddev=0.2, mean=0, seed=seed), name="v_%s" % i)
        b = tf.Variable(tf.truncated_normal([rank, 1], stddev=0.2, mean=0, seed=seed), name="b_%s" % i)
        Vx = tf.nn.relu(tf.matmul(V, Vx) + b)

    return Vx


def factorize(observed_features,
              labels,
              observed_features_validation,
              labels_validation,
              rank,
              max_iter=100,
              verbose=False,
              lambda_k=0,
              lambda_w=0,
              lambda_constants=0,
              epsilon=0.0001,
              optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
              depth=3,
              seed=12345):
    # Extract info about shapes etc from the training data
    num_items = observed_features.shape[0]
    num_features = observed_features.shape[1]

    # matrix defining the inner product weights when doing interactions
    K = tf.Variable(tf.truncated_normal([rank, rank], stddev=0.2, mean=0, seed=seed), name="metric_matrix")
    #10x10
    # coefficients for linear function on inputs (wide part)
    w = tf.Variable(tf.truncated_normal([1, num_features], stddev=0.2, mean=0, seed=seed), name="hyperplane")
    #1x5
    # coefficients for linear functinos on inputs (deep part)
    lw = tf.Variable(tf.truncated_normal([1, rank], stddev=0.2, mean=0, seed=seed), name="latenthyperplane")
    #1x10
    # bias in linear function
    b = tf.Variable(tf.truncated_normal([1, 1], stddev=0.2, mean=0, seed=seed), name="b_one")
    #1x1
    x = tf.placeholder(tf.float32, [None, num_features])
    y = tf.placeholder(tf.float32)

    norm_x = tf.nn.l2_normalize(x, dim=0)
    #999x5
    Vx = make_embeddings(tf.transpose(norm_x), rank, num_features, depth=depth, seed=seed)
    #10x999
    right_kern = tf.matmul(K, Vx)
    #10x999
    full_kern = tf.matmul(tf.transpose(Vx), right_kern)
    #999x999
    linear = tf.matmul(w, tf.transpose(norm_x))
    #1x999
    latent_linear = tf.matmul(lw, Vx)
    #1x999

    pred = tf.reduce_sum(tf.sigmoid(linear + latent_linear + full_kern + b))
    #999
    # todo: dropout. currently no regularization on the interaction layers in the cost functino
    # can handle with FTRL optimization
    cost = tf.reduce_mean(-y * tf.log(pred + 0.0000000001) - (1 - y) * tf.log((1 - pred + 0.0000000001)) +
                          lambda_k * tf.nn.l2_loss(K) +
                          lambda_w * tf.nn.l2_loss(w) +
                          lambda_constants * tf.nn.l2_loss(b))
    optimize = optimizer.minimize(cost)
    norm = tf.reduce_mean(tf.nn.l2_loss(w))
    saver = tf.train.Saver()
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        last_cost = 1000000
        for iter in range(0, max_iter):
            avg_cost = 0

            for i in range(num_items):
                _, c, n = sess.run([optimize, cost, norm],
                                   feed_dict={x: observed_features[i].reshape(1, num_features), y: labels[i]})
                avg_cost += c / num_items
            if verbose:
                print("epoch: %s, cost: %s" % (iter + 1, avg_cost))

            # check for convergence
            if abs(avg_cost - last_cost) / avg_cost < epsilon:
                break

            last_cost = avg_cost

        if verbose:
            print("optimization finished")
        predictions = []
        total_costs = 0
        for i in range(observed_features_validation.shape[0]):
            p, c = sess.run([pred, cost], feed_dict={x: observed_features_validation[i].reshape(1, num_features),
                                                     y: labels_validation[i]})
            predictions.append(p)
            total_costs += c
        saver.save(sess, 'model/deep_comp')
        #out_pred = sess.run(pred, feed_dict={x:get_data_out()})
        #print out_pred
        return predictions, total_costs / observed_features_validation.shape[0], sess.run([norm])


def predict_probab(ids, X_test, rank, seed, depth):
    num_features = X_test.shape[1]

    K = tf.Variable(tf.truncated_normal([rank, rank], stddev=0.2, mean=0, seed=seed), name="metric_matrix")
    w = tf.Variable(tf.truncated_normal([1, num_features], stddev=0.2, mean=0, seed=seed), name="hyperplane")
    lw = tf.Variable(tf.truncated_normal([1, rank], stddev=0.2, mean=0, seed=seed), name="latenthyperplane")
    b = tf.Variable(tf.truncated_normal([1, 1], stddev=0.2, mean=0, seed=seed), name="b_one")
    x = tf.placeholder(tf.float32, [None, num_features])
    y = tf.placeholder(tf.float32)

    norm_x = tf.nn.l2_normalize(x, dim=0)
    Vx = make_embeddings(tf.transpose(norm_x), rank, num_features, depth=depth, seed=seed)
    right_kern = tf.matmul(K, Vx)
    full_kern = tf.matmul(tf.transpose(Vx), right_kern)
    linear = tf.matmul(w, tf.transpose(norm_x))
    latent_linear = tf.matmul(lw, Vx)

    pred = tf.reduce_sum(tf.sigmoid(linear + latent_linear + full_kern + b))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        out_pred = []
        saver.restore(sess, 'model/deep_comp')
        for i in range(X_test.shape[0]):
            p= sess.run(pred, feed_dict={x: X_test[i].reshape(1, num_features),y: 1})
            out_pred.append(round(p,3))
        #print out_pred
        submit = pd.DataFrame({'ID': ids, 'click': out_pred})
        submit.to_csv('out.csv', index=False)

if __name__ == '__main__':
    X_train, X_valid, Y_train, Y_valid = get_data_vect('/home/mohit/comp_data/train2.csv')
    r = 15
    print "training started"
    predictions, test_costs, norm = factorize(X_train, Y_train, X_valid, Y_valid, r, verbose=True, depth=5)
    print("rank: %s, cost: %s, overall AUC: %s, norm: %s") % (
    r, test_costs, roc_auc_score(Y_valid, predictions, average="weighted"), norm)

    #print confusion_matrix(Y_valid, predictions)
    #ids, X = get_data_vect_out();
    #predict_probab(ids,X,r,12345,5)