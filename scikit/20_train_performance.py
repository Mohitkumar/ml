from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import numpy as np

categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']

train = fetch_20newsgroups(data_home='.', subset='train', shuffle=True,random_state=42, categories=categories)
test = fetch_20newsgroups(data_home='.', subset='test', shuffle=True,random_state=42, categories=categories)


def classify_text_nb():
    clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    clf.fit(train.data, train.target)
    return clf


def classify_text_svm():
   clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                   ('clf', SGDClassifier(loss='hinge', penalty='l2',random_state=42,alpha=1e-3,n_iter=5))])
   clf.fit(train.data, train.target)
   return clf

def predict_test_data(clf):
    print cross_val_score(clf, test.data, test.target, cv=3, scoring='accuracy')
    predicted = clf.predict(test.data)
    print confusion_matrix(y_true=test.target, y_pred=predicted)
    print np.mean(predicted == test.target)
    print metrics.classification_report(test.target, predicted,target_names=test.target_names)

if __name__ == '__main__':
    clf = classify_text_nb()
    predict_test_data(clf)
    print '##################'
    clf = classify_text_svm()
    predict_test_data(clf)