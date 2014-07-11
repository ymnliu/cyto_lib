#!/usr/bin/env python

import os
from os import listdir
from os import mkdir
from os import walk

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
f = []
i = 0

#img_dir = '/home/sijialiu/Dropbox/2014.2.21-Disomy-control-images'
img_dir = '/home/sijialiu/Dropbox/2014.2.21-Disomy-control-images'
output_dir = '../output_img'
sample_size = 1000

display = False

for (dirpath, dirnames, filenames) in walk(img_dir):
    f.extend(filenames[:sample_size])
    break


if display:
    cv2.imshow('image', im_f32)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if not os.path.exists(output_dir):
    mkdir(output_dir)

for img in f:
    im_uint16 = cv2.imread(img_dir + '/' + img, cv2.CV_LOAD_IMAGE_UNCHANGED)     
    im_f32 = np.float32(im_uint16) / 255 
    im_uint8 = im_uint16 / 255
    output_path_tmp =  output_dir + '/' + str(i).zfill(4) + '_' + img
    output_path_tmp = output_path_tmp.rsplit('.', 1)[0] + '.png'
    
    print str(i) + ' -> ' + output_path_tmp
    #cv2.imwrite(output_path_tmp, im_f32)
    imsave(output_path_tmp, im_uint16)

    if False:
	cv2.imshow('image', im_f32)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

    if False:
	plt.imshow(im_uint16)
	plt.show()

    i += 1

