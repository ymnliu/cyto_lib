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

def remove_all(dir_path):
    if os.path.exists(dir_path):
	shutil.rmtree(dir_path)		
	print "Directory files removed: " + dir_path
    else:
	print "Directory: \'" + dir_path + "\' does not exist"

def check_exist_and_create(dir_path):
    if not os.path.exists(dir_path):
	mkdir(dir_path)
	print 'Directory Created: ' + dir_path

f = []
#img_dir = '../data/monosomy_05082014'
#img_dir = '../data/trisomy_05082014'
img_dir = '../data/disomy_05082014'

output_dir = '../output_disomy'

remove_all(output_dir)
check_exist_and_create(output_dir)

sample_size = 200
print_flag = True

for (dirpath, dirnames, filenames) in walk(img_dir):
    f.extend(filenames)
    break

processed_count = 0

for idx, filename in enumerate(f):
    img_path = img_dir + '/' + filename
    #print 'opening: ' + img_path
    im16_org = ot.open_cyto_tiff(img_path)
    
    if im16_org is None:
	print "skip image: " + img_path 
	continue
    else:
	processed_count = processed_count + 1
    
    if processed_count > sample_size:
	break
    
    if print_flag:
	output_path = output_dir +'/' + \
		filename.rsplit('.', 2)[0] + '_org.png'
	print str(idx) + ": "+ output_path
	imsave(output_path, im16_org)
