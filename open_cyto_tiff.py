import os
from os import listdir
from os import mkdir
from os import walk

import cv2
import numpy as np
import skimage.io

def open_cyto_tiff(path):
    try:
	#im16 = cv2.cv.LoadImage(path, cv2.CV_LOAD_IMAGE_UNCHANGED)   
	#im_m16 = np.matrix(im16)
	#im_array = np.asarray(im16[:,:])
	im16 = skimage.io.imread(path, plugin='freeimage')
	return im16
    except:
	#print "skip image: " + path
	return None


