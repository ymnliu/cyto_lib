__author__ = 'sijialiu'

import time

import numpy as np

import util as cu
from image import CytoImage
import cyto.feature as cf
from cyto.util import print_block


label_list = [0, 1, 2]
file_list = [[], [], []]

for label in label_list:
    file_list[label] = cu.load_cyto_list(label)


def load_single_image(_label, idx):
    # print (label, idx)
    label = _label
    spot_feature = np.array([])

    file_name = file_list[label][idx]
    img = CytoImage(file_name)

    # img.cyto_show()
    img.get_cyto_spots()
    spots = img.spots

    return img


def prepare_train_data(train_size=100):
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


