#!/usr/bin/env python

from pyclass import train_test
from cyto.helper import prepare_train_data

########################################
##  training
##########################################

train, train_truth = prepare_train_data(100)
train_test.classifier_scores(train, train_truth)