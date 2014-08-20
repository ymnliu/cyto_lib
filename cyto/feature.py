from util import open_cyto_tiff
from util import load_cyto_list, serialize

import os
import sys
import shutil
from os import listdir
from os import mkdir
from os import walk
from scipy import misc 

from skimage import transform
from skimage import filter
from skimage.morphology import erosion, dilation, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk

from sklearn import mixture

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import imread, imsave
from scipy import ndimage


thres_ratio = .3
noise_thres_ratio = .5 * thres_ratio

def normalize_columns(arr):
    rows, cols = arr.shape
    for col in xrange(cols):
	arr[:,col] /= abs(arr[:,col]).max()

def pre_processing(img):
    return transform.downscale_local_mean(img, (2, 2))

def get_image_feature(img):
    im16 = img
    h, w = im16.shape
    max_idx = im16.argmax()

    top_hat = ndimage.morphology.white_tophat(im16, (5, 5))
    
    im16_blur = ndimage.gaussian_filter(top_hat, sigma=1)
    
    filter_org = im16_blur 
    
    peak = filter_org.max()
    mask =  filter_org > thres_ratio * peak 
   
    # ostu threshold
    #T = mahotas.thresholding.otsu(im16)
    T = filter.threshold_otsu(im16)

    mmask = im16 > thres_ratio * T
    im_ma = im16 * mmask
    
    labeled, nr_obj = ndimage.label(mask)
    max_label = labeled[max_idx / w][max_idx % w]
    max_mask = labeled == max_label 

    top_hat_mask = ndimage.morphology.white_tophat(mask, (4, 4))
    
    selem = disk(3)
    dmask = dilation(max_mask,selem)
    ndmask = ~dmask
    
    peak_template = dmask * im16
    removed_peak = ndmask * im16
    
    ratio_peak = removed_peak.max() / (peak * 1.)
    #aic, bic = get_aic_bic(img)
    #aic, bic = (0, 0) 
    

    feature_list = [T, ratio_peak, nr_obj, im16.max(), 
	    np.sum(im_ma), np.sum(im16)]
    
    return feature_list

feature_dsp = ['Otsu_threshold', 'peak ratio', 'detect number', 
	'peak value', 'square sum', 'intensity sum']


def get_features(img_dir, label, sample_len):
    'depreciate method '
    'use load_features instead.'

    f = []
    if not os.path.exists(img_dir):
	sys.exit( 'Dir cannot found: ' + img_dir )

    print 'searching the directory: ' + img_dir  

    for (dirpath, dirnames, filenames) in walk(img_dir):
	f.extend(filenames)
	break

    f.sort()
    if sample_len > 0 :
	sample_size = sample_len
    else:
	sample_size = len(f) 

    display = False #True 
    save_org_flag = False 
    verbose_flag = False

    if verbose_flag:
	print 'samples: ' + str(sample_size)

    skip_count = 0
    processed_count = 0
    
    feature_all = np.array([])

    for index in range(0, sample_size):
	img_path = img_dir + '/' + f[index]
	#print 'opening: ' + img_path
	im16_org = open_cyto_tiff(img_path)
	
	if im16_org is None:
	    skip_count = skip_count + 1
	    if verbose_flag:
	       print "skip image: " + path
	    continue
	else:
	    processed_count = processed_count + 1
	
	feature_tmp = get_image_feature(im16_org)
	 
	if feature_all.shape[0] is 0:
	    feature_all = np.asarray(feature_tmp)
	else:
	    feature_all = np.vstack((feature_all, feature_tmp))
    
    return feature_all


def load_features(label, sample_len):
    f = load_cyto_list(label)[:sample_len]

    feature_all = np.array([])

    for img_path in f:
	im16_org = open_cyto_tiff(img_path)
	
	feature_tmp = get_image_feature(im16_org)
	 
	if feature_all.shape[0] is 0:
	    feature_all = np.asarray(feature_tmp)
	else:
	    feature_all = np.vstack((feature_all, feature_tmp))
    
    return feature_all

def get_aic_bic(img):
    subplot_data = img

    ctype = ['spherical', 'tied', 'diag', 'full']

    aic_res = np.zeros(5)
    bic_res = np.zeros(5)

    res_idx = 0

    for ncomponents in range(1, 6):

	X_train = np.ceil(serialize(subplot_data))
	#clf = mixture.DPGMM(n_components=3, covariance_type='full')
	clf = mixture.GMM(n_components=ncomponents, covariance_type=ctype[2],
		n_iter=100)
	clf.fit(X_train)

	aic_res[res_idx] = clf.aic(X_train)
	bic_res[res_idx] = clf.bic(X_train)
	#print clf.weights_
	#print clf.means_
	#print clf.get_params()
#	x = np.linspace(0.0, subplot_data.shape[0])
#	y = np.linspace(0.0, subplot_data.shape[1])
#	X, Y = np.meshgrid(x, y)
#	XX = np.c_[X.ravel(), Y.ravel()]
#	#Z = np.log(-clf.score_samples(XX)[0])
#	Z = np.log(-clf.score_samples(XX)[0])
#	Z = Z.reshape(X.shape)

	res_idx += 1
    
    return np.argmin(aic_res) , np.argmin(bic_res)  
    #return np.concatenate(aic_res, bic_res)
