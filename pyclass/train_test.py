from sklearn import ensemble
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import numpy as np


def classifier_scores(train, train_truth, nfolds=10):
    res_mean = []
    res_std = []

    clf_list = [('AdaBoost', ensemble.AdaBoostClassifier(n_estimators=100)),
                ('Random Forest', ensemble.RandomForestClassifier()),
                ('SVM', svm.LinearSVC()),
                ('DT', tree.DecisionTreeClassifier()),
                ('GaussianNB', GaussianNB())]

    classifier_func_score(clf_list, train, train_truth, nfolds)


"""
    print "Mean:"
    print "%.2f, "*len(res_mean) % tuple(res_mean)

    print "Std"
    print "%.2f, "*len(res_std) % tuple(res_std)
"""


def show_confusion_matrix(X, y):
    """docstring for show_confusion_matrix"""

    print "show matrix..."
    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=.1)

    print "running classifier...."
    # Run classifier
    classifier = svm.SVC()
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    print "compute confusion matrix..."
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    cm_sum = np.sum(cm, axis=1).T

    cm_ratio = cm / cm_sum.astype(float)[:, np.newaxis]

    print(cm_ratio)
    print cm
    print cm_sum

    print "plot matrix..."
    # Show confusion matrix in a separate window
    plt.matshow(cm_ratio)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def classification_eval(train, train_truth, nfolds=3):
    classifier_scores(train, train_truth, nfolds)
    show_confusion_matrix(train, train_truth)


def classifier_func_score(_clf_list, train, train_truth, nfolds):
    for (desp, clf) in _clf_list:
        scores = cross_val_score(clf, train, train_truth, cv=nfolds)
        print("Accuracy %s: %0.2f (+/- %0.2f)" % (desp, scores.mean(), scores.std()))