import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
from skimage.measure import regionprops, label
from gaussian_func import gaussian_fit_2d
from cauchy_func import cauthy_fit_2d
from scipy import sparse


class CytoSpot:
    """
        class of spots in cytometry image
    """
    org_data = []
    data = []
    norm_data = []
    expand_size = 1  # add boundary expansion
    region = None
    origin = (-1, -1)
    size = (-1, -1)
    bbox = (-1, -1, -1, -1)
    noise_ratio = .1

    def __init__(self, data):
        org_data = data
        (x, y) = org_data.shape

        self.size = (x + 2 * self.expand_size, y + 2 * self.expand_size)

        self.data = np.zeros(self.size)

        self.data[self.expand_size:x + self.expand_size,
        self.expand_size:y + self.expand_size] = org_data

        self.norm_data = self.data / self.data.sum()
        # print self.data

        mask = self.data > self.noise_ratio * self.data.max()

        labeled_img = label(mask)

        regions = regionprops(labeled_img)
        self.region = regions[0]

    def get_features(self):
        region = self.region

        f = []
        minr, minc, maxr, maxc = region.bbox

        mat = self.data[minr: maxr, minc: maxc]
        width = maxc - minc
        height = maxr - minr

        f.append(region.area)
        f.append(region.extent)  # height width ratio
        f.append(region.convex_area)
        f.append(region.eccentricity)
        f.append(region.equivalent_diameter)

        f.append(region.solidity)  # Ratio of pixels in the
        # region to pixels of the convex hull image.

        f.append(mat.sum())
        f.append((mat ** 2).sum() / (region.area * 1.))
        f.append(mat.sum() / (region.area * 1.))

        try:
            pcov = self.get_gaussian_cov()
            # print self.get_2d_gaussian_param()[0].shape
            if pcov > 1e5:
                return None
                # from param idx 9-15
            f = np.concatenate((f, self.get_2d_gaussian_param()[0], np.array([pcov])))
            return f
        except:
            print "spot discarded"
            return None

    def get_gaussian_cov(self):
        popt, pcov = self.get_2d_gaussian_param()
        return np.sqrt(np.diag(pcov)).sum()

    def get_2d_gaussian_param(self):
        return gaussian_fit_2d(self.data)

    def get_2d_cauthy_param(self):
        return cauthy_fit_2d(self.data)

    def show_spot(self, ax):
        ax.plot(self.data)
        return ax

    def get_sparse_data(self):
        return sparse.coo_matrix(self.data)

    def get_log_data(self):
        return np.log(self.data + np.ones(self.data.shape))

