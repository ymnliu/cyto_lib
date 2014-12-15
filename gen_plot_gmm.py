import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

import  load_single_image as ls
from cyto.util import serialize
from cyto.feature import get_aic_bic_res
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
    #subplot_data = imgdata.T

    root = "../spot_plot/"
    n_components = 2

    ctype = ['spherical', 'tied', 'diag', 'full']

    X_train = np.floor(serialize(subplot_data))

    X = X_train

    # Fit a mixture of Gaussians with EM using five components
    gmm = mixture.GMM(n_components=n_components, covariance_type='full')
    gmm.fit(X)

    # Fit a Dirichlet process mixture of Gaussians using five components
    dpgmm = mixture.DPGMM(n_components=n_components, covariance_type=ctype[0], alpha=5.12, 
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
        fig = plt.figure()
        ax = fig.add_subplot(111)
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
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)

    #    plt.xlim(-10, 10)
    #    plt.ylim(-3, 6)
        plt.xticks(())
        plt.yticks(())
        #plt.title(title)
        str_name = "scatter_%d_%d_%d.eps" % (label, idx, spot_idx)
        fname = root + str_name
        plt.savefig(fname)
        
        print title
        logprob, responsibility = clf.score_samples(X)
        print logprob 
        print ""

        for j, X_sample in enumerate(X):
            x = int(X_sample[0])
            y = int(X_sample[1])

            log_mat[x, y] = logprob[j] * -1.
            #rsp_mat[x, y] = responsibility[j][0] / responsibility[j][1] 
            rsp_mat[x, y] = np.abs(responsibility[j][0] - responsibility[j][1] )
            #rsp_mat[x, y] = np.std(responsibility, axis=1 )

    #    print log_mat
       # splot = plt.subplot(s_nrows, s_ncols, 2 + s_i)
        fig = plt.figure()
        imgplot = plt.imshow(log_mat)
        imgplot.set_interpolation('nearest')
        plt.colorbar()
        plt.xticks(())
        plt.yticks(())
        str_name = "log_mat_%d_%d_%d.eps" % (label, idx, spot_idx)
        fname = root + str_name
        plt.savefig(fname)

        ### rsp mat ###########
        #fig = plt.figure()
        fig, ax = plt.subplots()
        cax = ax.imshow(rsp_mat, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
        rmax = rsp_mat.max()
        rmin = rsp_mat.min()
        rmed = (rmax + rmin) / 2.
        cbar = fig.colorbar(cax, ticks=[rsp_mat.min(), rmed,   rsp_mat.max()])
        
        str_name = "rsp_mat_%d_%d_%d.eps" % (label, idx, spot_idx)

        fname = root + str_name
        plt.tight_layout()
        plt.savefig(fname)
        #plt.show()
        

    fig = plt.figure()
    ax = plt.subplot(s_nrows, s_ncols, 4)
    cax = ax.imshow(subplot_data, interpolation='none',cmap='gray')
    
    plt.xticks(())
    plt.yticks(())
    cbar = fig.colorbar(cax, ticks=[subplot_data.min(), subplot_data.max()])
    
    #splot = plt.subplot(s_nrows, s_ncols, 8)
    #plt.imshow(imgdata)
    #dir_prefix = ["../non_overlap_view", "../overlap_view"]
    #fname = dir_prefix[category] + "/4by2_%d_%d_%d.eps" % sid
    #plt.savefig(fname)
    str_name = "spot_%d_%d_%d.eps" % (label, idx, spot_idx)
    fname = root + str_name
    plt.savefig(fname)
    
    n_bins =  16
    normed = 0 

    log_bins, lbin = np.histogram(log_mat, bins=n_bins, density=False)
    rsp_bins, rbin = np.histogram(rsp_mat, bins=n_bins, density=False)

    fig = plt.figure()
    #n, bins, patches = plt.hist(rsp_mat, n_bins, normed=normed, facecolor='blue', alpha=0.5)
    ax = fig.add_subplot(2,1,1)
    ax.plot(log_bins[2:], marker="o")
    ax = fig.add_subplot(2,1,2)
    ax.plot(rsp_bins[2:], marker="o")

    str_name = "bin_plot_%d_log_%d_%d_%d.eps" % (n_bins, label, idx, spot_idx)
    fname = root + str_name
    plt.savefig(fname)
    
    """
    str_name = "hist_%d_rsp_%d_%d_%d.eps" % (n_bins, label, idx, spot_idx)
    fname = root + str_name
    plt.savefig(fname)
   
    fig = plt.figure()
    n, bins, patches = plt.hist(log_mat, n_bins, normed=normed, facecolor='green', alpha=0.5)

    str_name = "hist_%d_log_%d_%d_%d.eps" % (n_bins, label, idx, spot_idx)
    fname = root + str_name
    plt.savefig(fname)"""
    
    return log_mat, rsp_mat, spot

def plot_beta_hist(a, b):
    plt.hist(beta(a, b, size=10000), histtype="stepfilled",
             bins=25, alpha=0.8, normed=True)
    return



sid = (2,45,0)
#sid = (0,10,0)
#sid = (1,8,0)
#sid = (0,20,0)
#sid = (0,30,0)
#sid = (1,329,0)
#sid = (1,330,0)
sid = (0,2,0)
sid = (0, 30, 0)
#sid = (2, 33, 1)
#sid = (2, 23, 0)
#sid = (1, 51, 0)
sid = (1, 8, 0)
log_mat, rsp_mat, spot = gen_set_gmm(sid, 0)

"""
n_bins = 10

log_bins, lbin = np.histogram(log_mat, bins=n_bins, density=True)
rsp_bins, rbin = np.histogram(rsp_mat, bins=n_bins, density=True)

fig = plt.figure()
n, bins, patches = plt.hist(rsp_mat.ravel(), n_bins, normed=1, facecolor='green', alpha=0.5)

plt.show()"""
