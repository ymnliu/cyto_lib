#!/usr/bin/env python

from cyto.image import CytoImage
from cyto.util import load_cyto_list

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

sample_count = 10
feature_all = []

#ax = fig.add_subplot(111)
color_str = ['rx', 'y^', 'bo']
fid = [0, 1, 6]

label_list = [0, 1, 2]

def load_single_image(label, idx):
    print (label, idx)

    file_list = load_cyto_list(label)
    spot_feature = np.array([])

    file_name = file_list[idx] 
    img = CytoImage(file_name)

    #img.cyto_show()
    img.get_cyto_spots()
    spots = img.spots

    return  img


"""
    display = True 




    #img.cyto_show()
    #spot_fig = img.show_each_spot()
    #spot_fig.suptitle(str(idx) +'\n' + img.path, fontsize=12)	
    #plt.show()

    for spot in spots:
	ftmp =  spot.get_features()
	#ftmp =  spot.get_2d_gaussian_param()
	sparse_matrix = spot.get_sparse_data()

	print "###" * 20

	if spot_feature.shape[0] is 0:
	    spot_feature = ftmp
	else:
	    spot_feature = np.vstack((spot_feature, ftmp))
	
    feature_all.append(spot_feature)

    



'''
#print [x for x in feature_all]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for label in label_list:
	color_idx = color_str[label]
	d2plot = feature_all[label].T
	#plt.plot(d2plot[0], d2plot[3], color_idx)

	ax.scatter(d2plot[fid[0]], d2plot[fid[1]],
		d2plot[fid[2]], c=color_idx[0],  marker=color_idx[1])

#ax.legend([p[0], p[1], p[2]], ["monosomy", "disomy", "trisomy"])
#plt.title('feature discrimination\n' + 
#	    str([cf.feature_dsp[x] for x in fid ]))
#print [cf.feature_dsp[x] for x in fid ]
#print fid
#ax.set_xlabel(cf.feature_dsp[fid[0]])
#ax.set_ylabel(cf.feature_dsp[fid[1]])
#ax.set_zlabel(cf.feature_dsp[fid[2]])

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
plt.show()

#print spot_feature

#ax = plt.subplot(2, 2, 1)
#img.cyto_show_spots(ax)
#img.show_each_spot()

#plt.imshow(img.get_mask())
#plt.show()
'''
"""
