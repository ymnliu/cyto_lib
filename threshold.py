#!/usr/bin/env python

import open_cyto_tiff as ot

import os
import sys
import shutil
from os import listdir
from os import mkdir
from os import walk
from scipy import misc 

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import imread, imsave
from scipy import ndimage

import mahotas
import pymorph as pm
import pylab

def remove_all(dir_path):
    if os.path.exists(dir_path):
	shutil.rmtree(dir_path)		
	print "Directory files removed: " + dir_path
    else:
	print "Directory: \'" + dir_path + "\' does not exist"

f = []
#img_dir = '/home/sijialiu/Dropbox/group/cytometry/canyouseethisimages'
#img_dir = '../data/monosomy_05082014'
#img_dir = '../data/trisomy_05082014'
img_dir = '../data/disomy_05082014'

if not os.path.exists(img_dir):
    sys.exit( 'Dir cannot found: ' + img_dir )

output_dir = '../output_img'
output_org = '../org_out'

remove_all(output_dir)
remove_all(output_org)

if not os.path.exists(output_dir):
    mkdir(output_dir)
    print 'Directory Created: ' + output_dir

if not os.path.exists(output_org):
    mkdir(output_org)
    print 'Directory Created: ' + output_org

print 'searching the directory: ' + img_dir  

for (dirpath, dirnames, filenames) in walk(img_dir):
    f.extend(filenames)
    break

f.sort()

sample_size = 40 #len(f) 

display = False #True 
save_org_flag = False 
verbose_flag = False


if sample_size < 20 :
    display = True

print 'samples: ' + str(sample_size)

result_list = []
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

    h, w = im16_org.shape
    max_idx = im16_org.argmax()

    #im16_org[im16_org < noise_thres_ratio * im16_org.max()] = 0

    im16 = im16_org
    #print im16.dtype
    #im_f32 = np.array(im16)

    #mm16 = np.mat(im16)
    #mmask = np.mat(mask)

    top_hat = ndimage.morphology.white_tophat(im16_org, (5, 5))
    
    im16_blur = ndimage.gaussian_filter(top_hat, sigma=1)
    
    filter_org = im16_blur 
    
    peak = filter_org.max()
    thres_ratio = .3
    noise_thres_ratio = .5 * thres_ratio
    mask =  filter_org > thres_ratio * peak 
   
    # ostu threshold
    T = mahotas.thresholding.otsu(im16)
    thres_list.append(T)
    mmask = im16 > thres_ratio * T
    im_ma = im16 * mmask
    
    #   pymorph
    labeled, nr_obj = ndimage.label(mask)
    max_label = labeled[max_idx / w][max_idx % w]
    max_mask = labeled == max_label 

    top_hat_mask = ndimage.morphology.white_tophat(mask, (4, 4))
    
    dmask = pm.dilate(max_mask)
    ndmask = ~dmask
    
    peak_template = dmask * im16_org
    removed_peak = ndmask * im16_org
    
    ratio_peak = removed_peak.max() / (peak * 1.)
    
    second_peak_list.append(ratio_peak)

    if False:  #ratio_peak > 0.9:
	plt.imshow(im16_org) 
	output_path = output_dir +'/peak_'  + str("%0.3f" % ratio_peak)+ \
		f[index].rsplit('.', 2)[0] + '.png'
	plt.savefig(output_path)

    #rmax = pm.regmax(im16)

#    if nr_obj > 3:
#	abnormal_list.append( f[index] )
#	display = True
    #im_mask = np.dot(mmask, mm16)
    # print T

    if peak > 2000:
	save_org_flag = True
    else: 
	save_org_flag = False

    save_org_flag = False
    print_flag = False

    if verbose_flag:
	print str(index) + ": " + f[index] + " \tResult: " + str(nr_obj) 

    result_list.append(nr_obj)
    peak_list.append(im16_org.max())
    
    if save_org_flag:
	output_path_org = output_org + '/'  + \
		f[index].rsplit('.', 2)[0] + '.png'
	#cv2.imwrite(output_path_org, im8)
	#misc.imsave(output_path_org, im8)
	#plt.plot(im16_org)
	#plt.colorbar()
	plt.imsave(output_path_org, im16_org)

    if display or print_flag: 
	fig = plt.figure(figsize=plt.figaspect(1.))
	plt.subplot(3,3,1)
	plt.imshow(im16_org)
	
	plt.subplot(3, 3, 2)
	plt.imshow(im16_blur)

	plt.subplot(3, 3, 3)
	#plt.imshow(mask)
	#plt.imshow(removed_peak)
	plt.imshow(top_hat)
	
	plt.subplot(3, 3, 4)
	plt.imshow(mask)
	
	plt.subplot(3,3,5)
	plt.imshow(top_hat_mask)
	#plt.subplot(2, 2, 4)
	#plt.imshow(im_ma)
	
	###########################################################
	###   3d plot
	###########################################################

	ax = fig.add_subplot(3, 3, 9, projection='3d')

	#plt.imshow(pymorph.overlay(im16, rmax))
	plt.title(f[index] + " \n num of label: " + str(nr_obj))
	
	X = np.arange(0, im16_org.shape[1])
	Y = np.arange(0, im16_org.shape[0])
	X, Y = np.meshgrid(X, Y)
	Z = im16_org

	#print Z.shape, Z.max()
	surf = ax.plot_surface(X, Y, Z, rstride=1, 
		cstride=1, linewidth=0, cmap=cm.coolwarm)
	
	#fig = plt.gcf()
	#fig.set_size_inches(18,10)
	if display:
	    plt.show()
	
	if print_flag:
	    output_path = output_dir +'/' + \
		    f[index].rsplit('.', 2)[0] + '_3_3.png'
	    print output_path
	    plt.savefig(output_path)
    
    energy_list.append(np.sum(im_ma))
    sys.stdout.write("Detection progress: %d%%   \r" % 
	    (100 * processed_count / (sample_size - skip_count) + 1) )
    sys.stdout.flush()
   
print ""
print ""
print "++" * 40
print "+ Result"
print "++" * 40
print ""

# print result_list  
# print peak_list
#print second_peak_list
# print "%0.3f  " * sample_size % tuple(second_peak_list)

np.append(result_list, peak_list, 2)
np.append(result_list, second_peak_list, 2)
np.append(result_list, energy_list, 2)
np.append(result_list, thres_list, 2)

#result_list.append(second_peak_list)

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
