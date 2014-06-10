#!/usr/bin/env python

import open_cyto_tiff as ot
import cyto_feature as cf
from yummy_util import print_block

from sklearn import svm
from sklearn.datasets import load_iris
from sklearn import ensemble

import numpy as np
import time


localtime = time.asctime( time.localtime(time.time()) )
print_block("Local current time :" +  localtime)


class_count = 3 
img_dir =[ '../data/monosomy_05082014',
	    '../data/disomy_05082014',
	    '../data/trisomy_05082014']

train_size = 30
test_size = 4 * train_size

train_real_size = np.zeros(class_count)
test_real_size = np.zeros(class_count)
train = np.array([])
test = np.array([])
train_truth = []
test_truth = []

# prepare training set
for i in 0, 1, 2:
    train_tmp = cf.get_features(img_dir[i], i + 1, train_size)
    train_real_size[i] = train_tmp.shape[0]
    train_truth_tmp =  (i + 1) * np.ones(train_real_size[i]) 
    train_truth =  np.hstack((train_truth, train_truth_tmp))

    if train.shape[0] is 0:
	train = train_tmp
    else:
	train = np.vstack((train, train_tmp))

# prepare test set
for i in 0, 1, 2:
    test_tmp = cf.get_features(img_dir[i], i + 1, test_size)
    test_real_size[i] = test_tmp.shape[0]
    test_truth_tmp =  (i + 1) * np.ones(test_real_size[i]) 
    test_truth =  np.hstack((test_truth, test_truth_tmp))

    if test.shape[0] is 0:
	test = test_tmp
    else:
	test = np.vstack((test, test_tmp))

#clf = tree.DecisionTreeClassifier()
clf = ensemble.RandomForestClassifier() 
clf = clf.fit( train , train_truth)

test_result = clf.predict(test)

#print test_result

error_count = [0, 0, 0]

error_matrix = np.zeros((3, 3))
error_rate_matrix = np.zeros((3, 3))

for i in range(0, len(test_truth)) :
    if test_result[i] != test_truth[i]:
	truth_idx = int(test_truth[i] - 1)
	error_idx = int(test_result[i] - 1)
	error_count[truth_idx] +=  1
	error_matrix[truth_idx][error_idx] += 1

#print str(error_count) + " out of " + str(len(test_truth))
#print error_count / (len(test_truth) * 1.)
print error_count    
print error_matrix

error_rate_matrix = error_matrix / (test_real_size - train_real_size).T
print error_rate_matrix

print_block("Test on Disomy Data")

test_dir = '../data/disomy_02212014'
test_tmp = cf.get_features(test_dir, 2, test_size)
test_result = clf.predict(test_tmp)

error_disomy_count = 0
for i in range(len(test_result)):
    if test_result[i] != 2:
	error_disomy_count += 1

print "Test Error Rate: " + str(error_disomy_count / (test_size * 1.))
