import warnings

import util as cu
import feature as cf
from spot import CytoSpot

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from skimage.morphology import skeletonize, dilation, disk
from skimage.measure import regionprops, label


class CytoImage:
    'Class of cytometry image'
    data = np.asarray([])
    noise_ratio = CytoSpot.noise_ratio

    def __init__(self, path):
        self.data = cu.open_cyto_tiff(path)
        self.path = path
        self.spots = []
        # self.data = self.preprocessing()
        self.get_cyto_spots()

    # print "spots count init"

    def preprocessing(self):
        masked_img = self.get_masked_img()
        return masked_img

    def get_masked_img(self):
        mask = self.get_mask()
        return mask * self.data

    def get_mask(self):
        img = self.data > self.noise_ratio * self.data.max()
        skeleton = skeletonize(img)
        mask = dilation(skeleton > 0, disk(2))
        return mask

    def get_cyto_spots(self):
        if len(self.spots) is 0:
            img = self.data
            labeled_img = label(self.get_mask())

            regions = regionprops(labeled_img)

            for region in regions:
                minr, minc, maxr, maxc = region.bbox
                mat = img[minr: maxr, minc: maxc]
                spot = CytoSpot(mat)
                spot.origin = (minr - spot.expand_size, minc - spot.expand_size)
                self.spots.append(spot)

        return self.spots

    def get_spots_feature(self):
        spots = self.get_cyto_spots()
        train = np.array([])

        for spot in spots:
            train_tmp = spot.get_features()
            if train_tmp is None:
                continue

            if train.shape[0] is 0:
                train = train_tmp
            else:
                print train.shape, train_tmp.shape
                train = np.vstack((train, train_tmp))

        if train.shape[0] is 1:
            train = train.T
        return train

    def get_feature(self):
        return np.array(cf.get_image_feature(self.data))


