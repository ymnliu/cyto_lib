__author__ = 'sijialiu'
import cyto.helper as ch
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
import matplotlib.patches as mpatches
import os
import cyto.util
import numpy as np
import cyto.gaussian_func
from scipy.stats import beta
import cyto.util

sample_size = 1000

path_prefix = "../../result"

spot_count_list = []

spot_max = np.zeros((3, sample_size, 5))

label_list = [0]

spot_cauthy_fit_feature = np.array([])

for label in label_list:
    for i in range(sample_size):
        img = ch.load_single_image(label, i)

        spots = img.get_cyto_spots()
        spot_count_list.append(len(spots))

        for spot_id, spot in enumerate(spots):
            # spot_max[label][i][spot_id] = np.max(spot.data)

            popt = spot.get_2d_cauthy_param()

            if isinstance(popt, basestring):
                # ch.save_spot(spot, cyto.util.root + "/../result/abnormal", popt)
                continue

            print "success"
            if spot_cauthy_fit_feature.shape[0] is 0:
                spot_cauthy_fit_feature = np.array(popt)
            else:
                spot_cauthy_fit_feature = \
                    np.vstack((spot_cauthy_fit_feature, popt))

    np.set_printoptions(precision=3)
    print spot_cauthy_fit_feature
    print spot_cauthy_fit_feature.shape

    sft = spot_cauthy_fit_feature.T

    for sf in sft:
        print ""
        print sf.std()
        print sf
