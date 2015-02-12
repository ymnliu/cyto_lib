import numpy as np
import matplotlib.pyplot as plt

import load_single_image as ls
from test.sample_list import overlap_list


n_sample = 5

non_overlap_list = [(1, i, 0) for i in range(n_sample)]

over_non_over = [non_overlap_list, overlap_list]

# labels = [0, 1]

labels = [0]

dir_prefix = ["../spot_org_view", "../spot_org_view"]

for category in labels:
    s_list = over_non_over[category][:n_sample]
    for sid in s_list:
        print sid
        label, idx, spot_idx = sid
        img = ls.load_single_image(label, idx)
        imgdata = img.data
        # img.show_each_spot()

        subplot_data = img.spots[spot_idx].data.T / np.float(np.max(img.spots[spot_idx].data.T))
        fname = dir_prefix[category] + "/org_%d_%d_%d.eps" % sid
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(subplot_data, cmap=plt.get_cmap('gray'), interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(fname)
        # plt.show()
        # plt.imshow(img, interpolation='nearest')
        #
        # plt.imsave(path, img, format='png')
        # fig, ax = plt.figure()
        # plt.imsave(fname, subplot_data, format='eps', cmap=plt.get_cmap('gray'))

