from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']

train = fetch_20newsgroups(data_home='.', subset='train', shuffle=True,random_state=42, categories=categories)

print len(train.data)
print len(train.target)


def classify_normal():
    count_vect = CountVectorizer()
    X_train_count = count_vect.fit_transform(train.data)
    print X_train_count.shape

    print count_vect.vocabulary_.get('advance')
    tf_idf_transformer = TfidfTransformer()
    X_train_tf = tf_idf_transformer.fit_transform(X_train_count)
    print X_train_tf.shape

    clf = MultinomialNB()
    clf.fit(X_train_tf, train.target)
    return clf, count_vect, tf_idf_transformer


def classify_pipeline():
    clf = Pipeline([('vect', CountVectorizer()),('tfidf',TfidfTransformer()), ('clf',MultinomialNB())])
    clf.fit(train.data, train.target)
    return clf


def predict(count_vect,tf_idf_transformer, clf, docs):
    X_pred_count = count_vect.transform(docs)
    X_pred_tf_idf = tf_idf_transformer.transform(X_pred_count)
    predicted = clf.predict(X_pred_tf_idf)
    for doc, category in zip(docs, predicted):
        print('%r => %s' % (doc, train.target_names[category]))


if __name__ == '__main__':
    clf, count_vect, tf_idf_transformer = classify_normal()
    docs_new = ['God is love', 'GPU is fast']
    predict(count_vect,tf_idf_transformer,clf,docs_new)

    pipe_clf = classify_pipeline()
    predicted = pipe_clf.predict(docs_new)
    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, train.target_names[category]))

