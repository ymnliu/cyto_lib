#!/usr/bin/env python
import open_cyto_tiff as ot
import cyto_local_feature as clf

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

def try_image(img_dir, file_list, index):
    filename = file_list[index]
    img_path = img_dir + '/' + filename
    print 'opening: ' + img_path
    img = ot.open_cyto_tiff(img_path)

    if img is None:
	print "skip image: " + img_path 
	return try_image(img_dir, file_list, index + 1)

    return img


img_dir = '../data/disomy_05082014'
file_list = []

for (dirpath, dirnames, filenames) in walk(img_dir):
    file_list.extend(filenames)
    break

processed_count = 0
idx = 100

    
im16_org = try_image(img_dir, file_list, idx)
print clf.get_local_feature(im16_org)

if True:
    plt.imshow(im16_org)
    plt.show()


