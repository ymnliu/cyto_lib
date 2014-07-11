#!/usr/bin/env python
import open_cyto_tiff as ot
import cyto_local_feature as clf
from cyto_image import CytoImage
from cyto_util import load_cyto_list

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

label = 2
processed_count = 0
idx = 110

file_list = load_cyto_list(label)

'''
for (dirpath, dirnames, filenames) in walk(img_dir):
    file_list.extend(filenames)
    break
'''

#im16_org, img_path = try_image(img_dir, file_list, idx)
#img_path = img_dir + '/' + file_list[idx]

#print clf.get_local_feature(im16_org)

img = CytoImage(file_list[idx])

#img.cyto_show()
#print img.get_spots_feature()
spots = img.get_cyto_spots()

for spot in spots:
    #print spot.get_features()

#ax = plt.subplot(2, 2, 1)
#img.cyto_show_spots(ax)
#img.show_each_spot()

#plt.imshow(img.get_mask())
#plt.show()
