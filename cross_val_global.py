#!/usr/bin/env python

import cyto.feature as cf
from cyto.util import print_block, load_cyto_list, open_cyto_tiff

from cyto.image import CytoImage

import train_test
import numpy as np
import time

localtime = time.asctime( time.localtime(time.time()) )
print_block("Local current time :" +  localtime)

class_count = 3 
img_dir =[ '../data/monosomy_05082014',
	    '../data/disomy_05082014',
	    '../data/trisomy_05082014']

train_size = 200

train_real_size = np.zeros(class_count)
train = np.array([])
test = np.array([])
train_truth = []
test_truth = []

print_block("Loading data")


# prepare training set
for label in 0, 1, 2:
    #train_tmp = cf.get_features(img_dir[label], label + 1, train_size)
    train_tmp = cf.load_features(label, train_size)
    train_real_size[label] = train_tmp.shape[0]
    train_truth_tmp =  (label + 1) * np.ones(train_real_size[label]) 
    train_truth =  np.hstack((train_truth, train_truth_tmp))

    if train.shape[0] is 0:
	train = train_tmp
    else:
	train = np.vstack((train, train_tmp))


######################################################################
##      Normailization
######################################################################

row_means = np.mean(train, axis=0)
#row_std = np.std(train, axis=0)
train_norm = (train / row_means) 

train = train_norm

########################################
##  training
##########################################

train_test.classification_val(train, train_truth)

