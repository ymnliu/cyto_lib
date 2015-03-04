#!/usr/bin/env python

from pyclass import train_test
from cyto.helper import prepare_train_data

########################################
##  training
##########################################

training_size = 10
print "training set size: " + str(training_size)

train, train_truth = prepare_train_data(training_size)

train_test.classifier_scores(train, train_truth)