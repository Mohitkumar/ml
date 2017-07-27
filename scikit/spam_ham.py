from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import numpy as np

data = load_files('spamham', categories=['ham', 'spam'], shuffle=True)

X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=.2)


def classify_text_svm():
   clf = Pipeline([('vect', CountVectorizer(decode_error='ignore')), ('tfidf', TfidfTransformer()),
                   ('clf', SGDClassifier(loss='hinge', penalty='l2',random_state=42,alpha=1e-3,n_iter=5))])
   clf.fit(X_train, y_train)
   return clf

def predict_data(clf):
    print cross_val_score(clf, X_test, y_test, cv=3, scoring='accuracy')
    predicted = clf.predict(X_test)
    print confusion_matrix(y_true=y_test, y_pred=predicted)
    print np.mean(predicted == y_test)
    print metrics.classification_report(y_test, predicted)

if __name__ == '__main__':
    clf = classify_text_svm()
    predict_data(clf)
