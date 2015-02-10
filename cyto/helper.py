__author__ = 'sijialiu'

import numpy as np

import util as cu
from image import CytoImage


label_list = [0, 1, 2]
file_list = [[], [], []]

for label in label_list:
    file_list[label] = cu.load_cyto_list(label)


def load_single_image(label, idx):
    # print (label, idx)

    spot_feature = np.array([])

    file_name = file_list[label][idx]
    img = CytoImage(file_name)

    # img.cyto_show()
    img.get_cyto_spots()
    spots = img.spots

    return img