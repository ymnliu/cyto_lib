#!/usr/bin/env python

import time

import numpy as np

from cyto.util import print_block, load_cyto_list
from cyto.image import CytoImage
from pyclass import train_test

localtime = time.asctime( time.localtime(time.time()) )
print_block("Local current time :" +  localtime)

class_count = 3 
img_dir =[ '../data/monosomy_05082014',
	    '../data/disomy_05082014',
	    '../data/trisomy_05082014']

train_size = 100
test_ratio = 2

test_size = int((test_ratio - 1) * train_size)
print test_size

train_real_size = np.zeros(class_count)
train_count = np.zeros(class_count)
test_real_size = np.zeros(class_count)
train = np.array([])
test = np.array([])
train_truth = []
test_truth = []

# prepare training set
for label in 0, 1, 2:
    #train_tmp = cf.get_features(img_dir[i], i + 1, train_size)
    file_list = load_cyto_list(label)
    
    for index in range(0, train_size):
	
	img_path = file_list[index]
	
	train_count[label] +=  1
	c_img = CytoImage(img_path)	
	try:
	    train_tmp = c_img.get_spots_feature()  
	    
	    if train.shape[0] is 0:
		train = train_tmp
	    else:
		train = np.vstack((train, train_tmp))
	
	except:
	    continue
	
	if train_tmp.ndim is 1:
	    row_count = 1
	else:
	    row_count = train_tmp.shape[0]

	train_truth_tmp = (label + 1) * np.ones(row_count) 
	train_truth =  np.hstack((train_truth, train_truth_tmp))

#train = np.nan_to_num(train)

########################################
##  training
##########################################

train_test.classification_val(train, train_truth)
