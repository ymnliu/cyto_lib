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

import load_single_image as ls
from cyto.util import serialize
from cyto.spot import CytoSpot


def unique_rows(data):
    uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
    return uniq.view(data.dtype).reshape(-1, data.shape[1])


def gen_set_gmm(sid, category):
    label, idx, spot_idx = sid
    img = ls.load_single_image(label, idx)

    img.show_each_spot()
    img.show_each_spot()

    imgdata = img.data
    spots = img.spots

    subplot_data = spots[spot_idx].data.T
    # subplot_data = imgdata.T

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

    X = unique_rows(X_train)
    #print "Unique rows"
    #print X

    s_nrows = 1
    s_ncols = 4

    #for s_i, (clf, title) in enumerate([(gmm, 'GMM'),
    #                                  (dpgmm, 'Dirichlet Process GMM')]):
    for s_i, (clf, title) in enumerate([(dpgmm, 'Dirichlet Process GMM')]):
        fig = plt.figure(figsize=(1, 4))
        splot = plt.subplot(s_nrows, s_ncols, 1 + s_i)
        Y_ = clf.predict(X)

        log_mat = np.zeros(subplot_data.shape)
        rsp_mat = np.zeros(subplot_data.shape)

        spot = CytoSpot(subplot_data)
        train_tmp = spot.get_features()

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

            # plt.xlim(-10, 10)
            # plt.ylim(-3, 6)
        plt.xticks(())
        plt.yticks(())
        #plt.title(title)

        plt.title("%d, %d, %d" % (label, idx, spot_idx))

        #print title
        logprob, responsibility = clf.score_samples(X)
        #print logprob 
        #print ""

        for j, X_sample in enumerate(X):
            x = int(X_sample[0])
            y = int(X_sample[1])

            log_mat[x, y] = logprob[j] * -1.
            #rsp_mat[x, y] = responsibility[j][0] / responsibility[j][1] 
            #rsp_mat[x, y] = np.abs(responsibility[j][0] - responsibility[j][1] )
            rsp_mat[x, y] = np.std(responsibility[j])

            # print log_mat
        splot = plt.subplot(s_nrows, s_ncols, 2 + s_i)
        imgplot = plt.imshow(log_mat)
        imgplot.set_interpolation('nearest')
        plt.colorbar()
        plt.xticks(())
        plt.yticks(())
        plt.title(title)

        splot = plt.subplot(s_nrows, s_ncols, 3 + s_i)
        imgplot = plt.imshow(rsp_mat)
        imgplot.set_interpolation('nearest')
        plt.colorbar()
        plt.xticks(())
        plt.yticks(())

    splot = plt.subplot(s_nrows, s_ncols, 4)
    plt.imshow(subplot_data.T, interpolation='none', cmap='gray')
    #plt.imshow(subplot_data)
    plt.xticks(())
    plt.yticks(())
    #splot = plt.subplot(s_nrows, s_ncols, 8)
    #plt.imshow(imgdata)
    dir_prefix = ["../non_overlap_view", "../overlap_view"]
    fname = dir_prefix[category] + "/4by2_%d_%d_%d.eps" % sid
    #plt.savefig(fname)
    #plt.show()

    return log_mat, rsp_mat, spot
