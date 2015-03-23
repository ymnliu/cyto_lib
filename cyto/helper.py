__author__ = 'sijialiu'

import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import util as cu
import cyto.feature as cf
from cyto.util import print_block
from image import CytoImage

label_list = [0, 1, 2]
file_list = [[], [], []]

for label in label_list:
    file_list[label] = cu.load_cyto_list(label)


def load_single_image(_label, idx):
    file_name = file_list[_label][idx]
    img = CytoImage(file_name)

    img.get_cyto_spots()

    return img


def prepare_train_data(train_size=50):
    localtime = time.asctime(time.localtime(time.time()))
    print_block("Local current time :" + localtime)
    class_count = 3
    train_real_size = np.zeros(class_count)
    train = np.array([])
    train_truth = []

    print_block("Loading data")
    # prepare training set
    for label in 0, 1, 2:
        # train_tmp = cf.get_features(img_dir[label], label + 1, train_size)
        train_tmp = cf.load_features(label, train_size)
        train_real_size[label] = train_tmp.shape[0]
        train_truth_tmp = (label + 1) * np.ones(train_real_size[label])
        train_truth = np.hstack((train_truth, train_truth_tmp))

        if train.shape[0] is 0:
            train = train_tmp
        else:
            train = np.vstack((train, train_tmp))

    return train, train_truth


"""
    ######################################################################
    ##      Normailization
    ######################################################################
    row_means = np.mean(train, axis=0)
    # row_std = np.std(train, axis=0)
    train_norm = (train / row_means)
    train = train_norm
"""


def show_each_spot(cyto_image):
    spots = cyto_image.spots
    subplot_num = int(max((len(spots) + 1), 2))
    fig, axarr = plt.subplots(subplot_num)

    idx = 0

    for spot in cyto_image.spots:
        axarr[idx].imshow(spot.data, interpolation='nearest')
        idx += 1

    axarr[idx].imshow(cyto_image.data, interpolation='nearest')

    # for spot in cyto_image.spots:
    # minr = (spot.origin[0] - spot.expand_size)
    #     maxr = spot.origin[0]
    #     minc = (spot.origin[1] - spot.expand_size)
    #     maxc = (spot.origin[1])
    #
    #     rect = mpatches.Rectangle((spot.origin[1], spot.origin[0]),
    #                               spot.size[1], spot.size[0],
    #                               fill=False, edgecolor='red', linewidth=1)
    #     axarr[idx].add_patch(rect)
    # plt.show()
    return fig


def cyto_show_spots(self, ax=None):
    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))

    ax.imshow(self.data)

    for spot in self.spots:
        minr = (spot.origin[0] - spot.expand_size)
        maxr = spot.origin[0]
        minc = (spot.origin[1] - spot.expand_size)
        maxc = (spot.origin[1])

        rect = mpatches.Rectangle((spot.origin[1], spot.origin[0]),
                                  spot.size[1], spot.size[0],
                                  fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    """
    Show how to modify the coordinate formatter to
    report the image "z"
    value of the nearest pixel given x and y
    """
    ax.format_coord = get_mat_value(self.data)

    # plt.show()
    return ax


def get_mat_value(mat):
    def format_coord(x, y):
        numrows, numcols = mat.shape
        col = int(x + 0.5)
        row = int(y + 0.5)
        if 0 <= col < numcols and 0 <= row < numrows:
            z = mat[row, col]
            return 'x=%1.2f, y=%1.2f, z=%1.2f' % (x, y, z)
        else:
            return 'x=%1.2f, y=%1.2f' % (x, y)

    return format_coord
