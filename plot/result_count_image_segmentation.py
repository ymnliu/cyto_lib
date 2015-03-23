import cyto.helper as ch
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
import matplotlib.patches as mpatches
import os
import cyto.util
import numpy as np


sample_size = 300

path_prefix = "../../result"

spot_count_list = []

spot_max = np.zeros((3, sample_size, 5))

for label in range(3):
    print "\nLabel %d" % label

    fig_dir = "{}/{}".format(path_prefix, label)

    try:
        os.makedirs(fig_dir)
    except OSError:
        if not os.path.isdir(fig_dir):
            raise

    for i in range(sample_size):
        img = ch.load_single_image(label, i)

        spots = img.get_cyto_spots()
        spot_count_list.append(len(spots))

        for spot_id, spot in enumerate(spots):
            spot_max[label][i][spot_id] = np.max(spot.data)
            # fig_path = "{}/{:04}.png".format(fig_dir, i)
            # print fig_path
            # ax = plt.subplot(111)
            # ch.cyto_show_spots(img, ax)
            # plt.savefig(fig_path)
            # plt.close()

    # count occurrences
    print "Label, Count -> Percentage"
    for i in range(1, 5):
        print "%d, %d  -> %.3f" % \
              (label, i,
               spot_count_list.count(i) / float(len(spot_count_list)))


