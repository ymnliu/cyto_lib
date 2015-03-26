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

sample_size = 300

path_prefix = "../../result"

spot_count_list = []

spot_max = np.zeros((3, sample_size, 5))

label_list = [0]

spot_gaussian_fit_feature = np.array([])

for label in label_list:
    for i in range(sample_size):
        img = ch.load_single_image(label, i)

        spots = img.get_cyto_spots()
        spot_count_list.append(len(spots))

        for spot_id, spot in enumerate(spots):
            # spot_max[label][i][spot_id] = np.max(spot.data)

            popt = spot.get_2d_gaussian_param()
            if popt is None:
                continue

            if spot_gaussian_fit_feature.shape[0] is 0:
                spot_gaussian_fit_feature = np.array(popt)
            else:
                spot_gaussian_fit_feature = \
                    np.vstack((spot_gaussian_fit_feature, popt))

    np.set_printoptions(precision=3)
    print spot_gaussian_fit_feature
    print spot_gaussian_fit_feature.shape

    sft = spot_gaussian_fit_feature.T

    for sf in sft:
        print ""
        print sf.mean()

        print sf.std()

