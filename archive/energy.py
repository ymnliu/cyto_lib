from os import walk

import cv2

import numpy as np
import matplotlib.pyplot as plt

f = []
i = 0

#img_dir = '../output_img'
img_dir = '/home/sijialiu/Dropbox/2014.2.21-Disomy-control-images'
sample_size = 2000

display = False

print 'searching the directory: ' + img_dir  

for (dirpath, dirnames, filenames) in walk(img_dir):
    f.extend(filenames)
    break

f = f[:sample_size]

if display:
    cv2.imshow('image', im_f32)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

sum_list = []
energy_list = []

print len(f)
for img in f:
    im_uint16 = cv2.imread(img_dir + '/' + img, cv2.CV_LOAD_IMAGE_UNCHANGED)    
    im_m16 = np.matrix(im_uint16)
#    print type(im_m16)
    sum_col = im_m16.sum()
    sq_im = im_uint16 ** 2
    im_m2 = np.matrix(sq_im)
    sum_sq = im_m2.sum()

    sum_list.append(sum_col)
    energy_list.append(sum_sq)

#   print 'col + energy'
#    print sum_col
#    print 'co    print sum_sq
#    print 'co    
#    print 'co    print "==" * 10
#    print 
    if False:
	cv2.imshow('image', im_f32)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

    if False:
	plt.imshow(im_uint16)
	plt.show()

    i += 1
    print str(i) + ' out of ' + str(len(f))
    

print '==' * 20
print 'ploting...'
#hist(energy_list, 10)
num_bins = 40
#n, bins, patches = plt.hist(sum_list, num_bins)
n, bins, patches = plt.hist(energy_list, num_bins)

plt.title(r'Histogram of energy of matrix')
plt.show()

