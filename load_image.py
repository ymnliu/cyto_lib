#!/usr/bin/env python
import open_cyto_tiff as ot
import cyto_local_feature as clf
from cyto_image import CytoImage


import os
import sys
import shutil
from os import listdir
from os import mkdir
from os import walk
from scipy import misc 

from cyto_image import CytoImage

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import imread, imsave
from scipy import ndimage

def try_image(img_dir, file_list, index):
    filename = file_list[index]
    img_path = img_dir + '/' + filename
    #print 'opening: ' + img_path
    img = ot.open_cyto_tiff(img_path)

    if img is None:
	print "skip image: " + img_path 
	return try_image(img_dir, file_list, index + 1)

    print 'opening: ' + img_path
    return img, img_path 


#img_dir = '../data/disomy_05082014'
img_dir = '../data/trisomy_05082014'
file_list = []

for (dirpath, dirnames, filenames) in walk(img_dir):
    file_list.extend(filenames)
    break

processed_count = 0
idx = 107

im16_org, img_path = try_image(img_dir, file_list, idx)
#img_path = img_dir + '/' + file_list[idx]

#print clf.get_local_feature(im16_org)
img = CytoImage(img_path)

img.cyto_show()
#print img.get_spots_feature()
spots = img.get_cyto_spots()

for spot in spots:
    print spot.get_features()

#ax = plt.subplot(2, 2, 1)
#img.cyto_show_spots(ax)
img.show_each_spot()

#plt.imshow(img.get_mask())
plt.show()
