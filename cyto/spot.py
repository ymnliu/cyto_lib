import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
from skimage.measure import regionprops, label
from gaussian_func import gaussian_fit_2d
from scipy import sparse


class CytoSpot:
    """
        class of spots in cytometry image
    """
    org_data = []
    data = []
    norm_data = []
    epd_sz = 1
    region = None
    origin = (-1, -1)
    size = (-1, -1)
    bbox = (-1, -1, -1, -1)
    noise_ratio = .3

    def __init__(self, data):
        org_data = data
        (x, y) = org_data.shape

        self.data = np.zeros((x + 2 * self.epd_sz,
                              y + 2 * self.epd_sz))

        self.size = (x + 2 * self.epd_sz, y + 2 * self.epd_sz)
        self.data[self.epd_sz:x + self.epd_sz,
        self.epd_sz:y + self.epd_sz] = org_data

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
        popt, pcov = gaussian_fit_2d(self.data)
        return popt, pcov

    def show_spot(self, ax):
        ax.plot(self.data)
        return ax

    def get_sparse_data(self):
        return sparse.coo_matrix(self.data)

    def get_log_data(self):
        return np.log(self.data + np.ones(self.data.shape))

        # def serilize(self):
        # data = np.floor(self.data/50)
        #     res = np.array([])
        #     it = np.nditer(data, flags=['multi_index'])
        #     while not it.finished:
        #         (d, x, y) =  (it[0], it.multi_index[0], it.multi_index[1])
        #         dtmp = np.tile([x, y], (d, 1))
        #         if res.shape[0]is 0:
        #         res = dtmp
        #         else:
        #         res = np.vstack((res, dtmp))
        #
        #         it.iternext()
        #
        #     return res
