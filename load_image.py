#!/usr/bin/env python

from cyto_image import CytoImage
from cyto_util import load_cyto_list

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

sample_count = 1000
feature_all = []

#ax = fig.add_subplot(111)
color_str = ['rx', 'y^', 'bo']
fid = [0, 1, -1]

label_list = [0, 1, 2]
fig_spot = plt.figure()
ax_spot = fig_spot.add_subplot(111)

feature_mean = []
feature_std = []

for label in label_list: 

    idx = 0
    file_list = load_cyto_list(label)
    spot_feature = np.array([])

    res = np.zeros(6) 

    for file_name in file_list[:sample_count] :
	img = CytoImage(file_name)

	#img.cyto_show()
	img.get_cyto_spots()
	spots = img.spots
	#print str(idx) + " -> " + str(len(spots))

	tmp_res = len(spots) 
	res[tmp_res] += 1

	display = False 

	if tmp_res != label + 1:
	    if display:
		#    img.cyto_show()
		spot_fig = img.show_each_spot()
		spot_fig.suptitle(str(idx) +'\n' + img.path, fontsize=12)	
		plt.show()

	for spot in spots:
	    ftmp =  spot.get_features()
	    if ftmp is None:
		continue

	    if spot_feature.shape[0] is 0:
		spot_feature = ftmp
	    else:
		spot_feature = np.vstack((spot_feature, ftmp))
	
	idx += 1
    

    feature_mean.append(np.mean(spot_feature, axis=0))

    feature_std.append( np.std(spot_feature, axis=0))
    
    if label is 0:
	mean_base = feature_mean
	std_base = feature_std


    feature_all.append(spot_feature)
    print res

print len(feature_mean)

for feature_vec in feature_mean:
    ax_spot.plot((feature_vec / mean_base).T)

    #ax_spot.plot(feature_std)
ax_spot.set_ylim((0, 2))
fig_spot.show()

    #feature_cov = np.cov(feature_all)
    #print feature_cov 

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
