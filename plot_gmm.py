"""
=================================
Gaussian Mixture Model Ellipsoids
=================================

Plot the confidence ellipsoids of a mixture of two Gaussians with EM
and variational Dirichlet process.

Both models have access to five components with which to fit the
data. Note that the EM model will necessarily use all five components
while the DP model will effectively only use as many as are needed for
a good fit. This is a property of the Dirichlet Process prior. Here we
can see that the EM model splits some components arbitrarily, because it
is trying to fit too many components, while the Dirichlet Process model
adapts it number of state automatically.

This example doesn't show it, as we're in a low-dimensional space, but
another advantage of the Dirichlet process model is that it can fit
full covariance matrices effectively even when there are less examples
per cluster than there are dimensions in the data, due to
regularization properties of the inference algorithm.
"""
import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

import  load_single_image as ls
from cyto.util import serialize
from cyto.feature import get_aic_bic_res

n_samples = 300
#label, idx = (2, 6, 0)
label, idx, spot_idx = (1, 8, 0)
#label, idx = (2, 22, 0)
label, idx, spot_idx = (2, 23, 0)
label, idx, spot_idx = (2, 33, 1)
#label, idx, spot_idx  = (0, 35, 0)
label, idx, spot_idx = (2, 40, 0)
label, idx, spot_idx = (2, 40, 1)
#label, idx, spot_idx = (1, 8, 0)
#label, idx, spot_idx = (2, 55, 0)

overlap_list = [(1, 8, 0),
                (2, 23, 0),
                (2, 33, 1),
                (0, 35, 0),
                (2, 40, 0),
                (1, 8, 0),
                (2, 55, 0)]

non_overlap_list = [(0,0,0),
                    (0,1,0),
                    (0,2,0),
                    (0,3,0),
                    (0,4,0),
                    (0,5,0)]

label, idx, spot_idx = overlap_list[2]
label, idx, spot_idx = non_overlap_list[1]


img = ls.load_single_image(label, idx)

img.show_each_spot()
img.show_each_spot()

imgdata = img.data
spots = img.spots

subplot_data = spots[spot_idx].data.T
#subplot_data = imgdata.T

n_components = 2

ctype = ['spherical', 'tied', 'diag', 'full']

X_train = np.floor(serialize(subplot_data))

X = X_train

# Fit a mixture of Gaussians with EM using five components
gmm = mixture.GMM(n_components=n_components, covariance_type='full')
gmm.fit(X)

# Fit a Dirichlet process mixture of Gaussians using five components
dpgmm = mixture.DPGMM(n_components=n_components, covariance_type=ctype[2], alpha=5.12, 
        params='wmc', n_iter=1000)
dpgmm.fit(X)

print dpgmm.n_components
color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

for i, (clf, title) in enumerate([(gmm, 'GMM'),
                                  (dpgmm, 'Dirichlet Process GMM')]):
    splot = plt.subplot(2, 1, 1 + i)
    Y_ = clf.predict(X)
    for i, (mean, covar, color) in enumerate(zip(
            clf.means_, clf._get_covars(), color_iter)):
        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 16.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

#    plt.xlim(-10, 10)
#    plt.ylim(-3, 6)
    plt.xticks(())
    plt.yticks(())
    #plt.title(title)

    plt.title("%d, %d, %d" % (label, idx, spot_idx))
    
    print title
    print clf.eval(X)
    print ""


plt.show()
