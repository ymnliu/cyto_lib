import open_cyto_tiff as ot

import os
import sys
import shutil
from os import listdir
from os import mkdir
from os import walk
from scipy import misc 

from skimage import transform
from skimage import filter
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk



import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import imread, imsave
from scipy import ndimage



def normalize_columns(arr):
    rows, cols = arr.shape
    for col in xrange(cols):
	arr[:,col] /= abs(arr[:,col]).max()

def pre_processing(img):
    return transform.downscale_local_mean(img, (2, 2))

def get_features(img_dir, label, sample_len):
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

    result_list = []
    obj_num_list = []
    peak_list = []
    second_peak_list = [] #np.zeros(sample_size)
    energy_list = []
    thres_list = []

    abnormal_list = []
    skip_count = 0
    processed_count = 0

    for index in range(0, sample_size):
	img_path = img_dir + '/' + f[index]
	#print 'opening: ' + img_path
	im16_org = ot.open_cyto_tiff(img_path)
	
	if im16_org is None:
	    skip_count = skip_count + 1
	    if verbose_flag:
	       print "skip image: " + path
	    continue
	else:
	    processed_count = processed_count + 1

	im16 = pre_processing(im16_org)
	#im16 = im16_org
	h, w = im16.shape
	max_idx = im16.argmax()

	#im16[im16 < noise_thres_ratio * im16.max()] = 0

	#print im16.dtype
	#im_f32 = np.array(im16)

	#mm16 = np.mat(im16)
	#mmask = np.mat(mask)

	top_hat = ndimage.morphology.white_tophat(im16, (5, 5))
	
	im16_blur = ndimage.gaussian_filter(top_hat, sigma=1)
	
	filter_org = im16_blur 
	
	peak = filter_org.max()
	thres_ratio = .3
	noise_thres_ratio = .5 * thres_ratio
	mask =  filter_org > thres_ratio * peak 
       
	# ostu threshold
	#T = mahotas.thresholding.otsu(im16)
	T = filter.threshold_otsu(im16)

	thres_list.append(T)
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
	
	second_peak_list.append(ratio_peak)

	result_list.append(label)
	obj_num_list.append(nr_obj)
	peak_list.append(im16.max())
	
	energy_list.append(np.sum(im_ma))
	#sys.stdout.write("Preparing features: %d%%   \r" % 
	#	(100 * processed_count / (sample_size - skip_count) + 1) )
	#sys.stdout.flush()
      
    result_array = np.asarray(result_list)
    obj_array = np.asarray(obj_num_list)
    peak_array = np.asarray(peak_list)
    second_peak_array = np.asarray(second_peak_list)
    energy_array = np.asarray(energy_list)
    thres_array  = np.asarray(thres_list)

    """
    np.append(result_array, obj_array)
    np.append(result_array, peak_array)
    np.append(result_array, second_peak_array)
    np.append(result_array, energy_array)
    np.append(result_array, thres_array)
    """
    result_all = [obj_array,
    	    peak_array, second_peak_array, energy_array, thres_array]
    #print result_all
    #result_all = [
    #	    peak_array, second_peak_array]

    #result_all = [obj_num_list, peak_list, second_peak_list,
	#    energy_list, thres_list]
    result_all = np.array(result_all)
    result_all = result_all.transpose()
    
    #normalize_columns(result_all)
    """
    print "proccessed cases: " + str(processed_count)
    print "skipped cases: " + str(skip_count)
    print "sample size: " + str(sample_size)
    print "processed ratio: " + "%.2f" % (processed_count / (sample_size * 1.))

    print ""
    print "++" * 40
    print ""

    #print energy_list
    print "thre_ratio = " + str(thres_ratio)

    for i in range(1, max(result_list) + 1):
	print str(i) + ': ' + str(result_list.count(i))

    if True:
	fig, ax = plt.subplots()
	num_bins = sample_size / 20
	if num_bins > 1:
	    n, bins, patches = plt.hist(peak_list, num_bins)
	    plt.title(r"Histogram of peak intensity:\\n" + img_dir)
	    plt.xlim(0, 4000)
	    plt.show()
	    """
    return result_all
