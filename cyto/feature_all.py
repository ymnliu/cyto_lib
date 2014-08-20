#!/usr/bin/env python

import open_cyto_tiff as ot
import cyto_feature as cf
from cyto_util import print_block, load_cyto_list

from cyto_image import CytoImage

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

########################################
##  training
##########################################

print_block("Training Classifier...")

#clf = tree.DecisionTreeClassifier()
clf = ensemble.RandomForestClassifier() 
clf = clf.fit( train , train_truth)

############################################################
##	Stream Testing
############################################################

testing_count = np.zeros((3, 1))
error_count = np.zeros((3, 1))

error_matrix = np.zeros((3, 3))
error_rate_matrix = np.zeros((3, 3))

for label in 0, 1, 2:
    file_list = load_cyto_list(label)
    
    for index in range(0, test_size):
	img_path = file_list[index]
	#print 'opening: ' + img_path
	#im16_org = ot.open_cyto_tiff(img_path)
	
	testing_count[label] +=  1
	c_img = CytoImage(img_path)	
	c_img.get_feature()	
	
	
	'''
	#feature_tmp = get_image_feature(im16_org)
	#res = clf.predict(c_img.get_feature())[0]
	
	if res != (label + 1):
	    truth_idx = int(label)
	    error_idx = int(res - 1)
	    error_count[truth_idx] +=  1
	    error_matrix[truth_idx][error_idx] += 1

	print str(label + 1) +  " -> " + str( res )

#print test_result

print error_count    
print error_matrix

error_rate_matrix = error_matrix /  (testing_count).T
print error_rate_matrix

print_block("Test on Disomy Data")

test_dir = '../data/disomy_02212014'
test_tmp = cf.get_features(test_dir, 2, test_size)
test_result = clf.predict(test_tmp)

error_disomy_count = 0
for i in range(len(test_result)):
    if test_result[i] != 2:
	error_disomy_count += 1

print test_result
print "Test Error Rate: " + str(error_disomy_count / (test_size * 1.))
'''
