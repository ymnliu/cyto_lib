#!/usr/bin/env python

import time
from random import shuffle

from sklearn import ensemble
import numpy as np

from cyto.util import print_block, load_cyto_list
from cyto.image import CytoImage


localtime = time.asctime( time.localtime(time.time()) )
print_block("Local current time :" +  localtime)

class_count = 3 
img_dir =[ '../data/monosomy_05082014',
	    '../data/disomy_05082014',
	    '../data/trisomy_05082014']

train_size = 500
test_ratio = 1.1

test_size = int((test_ratio - 1) * train_size)
print test_size

train_real_size = np.zeros(class_count)
test_real_size = np.zeros(class_count)
train = np.array([])
test = np.array([])
train_truth = np.array([])
test_truth = []

# prepare training set
'''
for i in 0, 1, 2:
    train_tmp = cf.get_features(img_dir[i], i + 1, train_size)
    train_real_size[i] = train_tmp.shape[0]
    train_truth_tmp =  (i + 1) * np.ones(train_real_size[i]) 
    train_truth =  np.hstack((train_truth, train_truth_tmp))

    if train.shape[0] is 0:
	train = train_tmp
    else:
	train = np.vstack((train, train_tmp))
'''

for label in 0, 1, 2:
    file_list = load_cyto_list(label)
    shuffle(file_list)
    
    for index in range(0, train_size):
	
	img_path = file_list[index]
	
	c_img = CytoImage(img_path)	
	train_tmp = c_img.get_feature()  
	
	if train.shape[0] is 0:
	    train = train_tmp
	else:
	    train = np.vstack((train, train_tmp))

	train_truth_tmp =  (label + 1) * np.ones(1) 
	train_truth =  np.hstack((train_truth, train_truth_tmp))

print train.shape
print train_truth.shape

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

hi_matrix = np.zeros((3, 3))

for label in 0, 1, 2:
    file_list = load_cyto_list(label)
    shuffle(file_list)

    for index in range(train_size,  train_size + test_size):
	img_path = file_list[index]
	#print 'opening: ' + img_path
	#im16_org = ot.open_cyto_tiff(img_path)
	
	testing_count[label] +=  1
	c_img = CytoImage(img_path)	
	#feature_tmp = get_image_feature(im16_org)
	res = clf.predict(c_img.get_feature())[0]
	
	if res != (label + 1):
	    truth_idx = int(label)
	    error_idx = int(res - 1)
	    error_count[truth_idx] +=  1
	    error_matrix[truth_idx][error_idx] += 1
	    cyto_count = int(c_img.get_feature()[2]) 
	    print  str(label + 1) + " -> " + str(cyto_count)
    
	    if label == 1 and res == 3: 
		hi_matrix[truth_idx][cyto_count-1] += 1
	    if label == 2 and res < 3: 
		hi_matrix[truth_idx][cyto_count -1] += 1

	#print str(label + 1) +  " -> " + str( res )

#print test_result

print error_count    
print error_matrix
print hi_matrix
error_rate_matrix = error_matrix /  (testing_count).T
print error_rate_matrix

