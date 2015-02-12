import time

import numpy as np

import cyto.feature as cf
from cyto.util import print_block, load_cyto_list
from cyto.image import CytoImage
from pyclass import train_test


localtime = time.asctime( time.localtime(time.time()) )
print_block("Local current time :" +  localtime)

class_count = 3 
img_dir =[ '../data/monosomy_05082014',
	    '../data/disomy_05082014',
	    '../data/trisomy_05082014']

train_size = 50

train_real_size = np.zeros(class_count)
train = np.array([])
test = np.array([])
train_truth = []
test_truth = []

print_block("Loading data")

res = np.zeros((3, 3))
spot_res = np.zeros((3, 3))

# prepare training set
for label in 0, 1, 2:
    #train_tmp = cf.get_features(img_dir[label], label + 1, train_size)

    sample_len = train_size
    f = load_cyto_list(label)[:sample_len]

    feature_all = np.array([])

    for img_path in f:
	#im16_org = open_cyto_tiff(img_path)
	
	c_img = CytoImage(img_path)	
	#train_tmp = c_img.get_spots_feature()  
        spots = c_img.get_cyto_spots()
        
        spot_count = 0
	
        for spot in spots:
            #feature_tmp = cf.get_gmm_selection(spot.data)
            feature_tmp = cf.get_gmm_matrix(spot.data)

            if feature_tmp.ndim is 1:
                spot_count += feature_tmp[0]
            else:
                spot_count += np.sum(feature_tmp)[0]
            
             
            #print feature_tmp
            if feature_all.shape[0] is 0:
                feature_all = np.asarray(feature_tmp)
            else:
                feature_all = np.vstack((feature_all, feature_tmp))
    
        if spot_count >= 3:
            spot_count = 3

        len_spots= len(spots)
         
        if len_spots >= 3:
            len_spots = 3

        res[label][spot_count - 1] += 1 
        spot_res[label][len_spots - 1] += 1 

    train_tmp  = feature_all

    #train_tmp = cf.load_features(label, train_size)
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

print res
print spot_res
