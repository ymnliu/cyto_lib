"""
=============================================
Density Estimation for a mixture of Gaussians
=============================================

Plot the density estimation of a mixture of two gaussians. Data is
generated from two gaussians with different centers and covariance
matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture

import load_single_image as ls
from cyto.util import serialize


n_samples = 300
# label, idx = (2, 6)
label, idx = (1, 8)
#label, idx = (2, 22)
#label, idx = (2, 23)
#label, idx = (2, 33)
#label, idx = (2, 40)
#label, idx = (2, 45)
#label, idx = (2, 55)
#label, idx = (2, 85)
#label, idx = (1, 8)
#label, idx = (2, 500)
label, idx = (0, 2)
label, idx = (0, 30)
#label, idx = (2, 23)
#label, idx = (1, 51)
#label, idx = (2, 33)

img = ls.load_single_image(label, idx)

img.show_each_spot()

imgdata = img.data
spots = img.spots

subplot_data = spots[0].data
#subplot_data = imgdata.T

n_components = 2

ctype = ['spherical', 'tied', 'diag', 'full']

X_train = np.floor(serialize(subplot_data))

#np.savetxt("subplot_2_8.txt", X_train, fmt='%s')

#X_train = (X_train - np.mean(X_train))/ np.mean(X_train)
# print "########################################"
# print X_train

clf = mixture.DPGMM(n_components=n_components, covariance_type=ctype[0], n_iter=200, alpha=5.,
                    verbose=False, random_state=20, params='wm', init_params='wmc', thresh=.00001)
#clf = mixture.VBGMM(n_components=ncomponents, covariance_type=ctype[3], n_iter=10, alpha=1., 
#        verbose=True, random_state=5, params='wmc', init_params='m')
#clf = mixture.GMM(n_components=ncomponents, covariance_type=ctype[3],
#	n_iter=100)
clf.fit(X_train)

print "Weights:"
print clf.weights_
print "Means: "
print clf.means_
print clf.get_params()
# print clf.covars_
#print clf.covars_.shape
print clf._get_covars()

#f = np.concatenate((clf.weights_.ravel(), clf.means_.ravel(), clf.precs_.ravel()))
f = np.concatenate((clf.weights_.ravel(), clf.means_.ravel()))

ncomponents = len(clf.weights_)

x = np.linspace(0.0, subplot_data.shape[0])
y = np.linspace(0.0, subplot_data.shape[1])
X, Y = np.meshgrid(x, y)
XX = np.c_[X.ravel(), Y.ravel()]
#Z = np.log(-clf.score_samples(XX)[0])
Z = -clf.score_samples(XX)[0]
print clf.score(XX)
Z = Z.reshape(X.shape)

#z = 0
#print clf.lower_bound(X, z)


fig = plt.figure()
#ax = fig.add_subplot(1, 2, 1, aspect='equal')
ax = fig.add_subplot(1, 1, 1)

#CS = ax.contour(X, Y, Z)
#CB = ax.colorbar(CS, shrink=0.8, extend='both')
#ax.scatter(X_train[:, 0], X_train[:, 1], .8)

truncate_ratio = 2
thres = 1 / float(truncate_ratio * ncomponents)  #0.5 / ncomponents

#thres = 0
print "%d compoents larger than %.2f" % (len(np.argwhere(clf.weights_ > thres)), thres)

m_weights = [.45, .55]

for cp_idx in range(0, ncomponents):
    if clf.weights_[cp_idx] > thres:
        #str_weight = "%.2f" % clf.weights_[cp_idx] 
        str_weight = "%.2f" % m_weights[cp_idx]
        ax.scatter(clf.means_[cp_idx, 0].T, clf.means_[cp_idx, 1].T, c='r', marker='o')
        ax.text(clf.means_[cp_idx, 0].T, clf.means_[cp_idx, 1].T, str_weight, color='r', fontsize=40)

CS = ax.contour(X, Y, Z)
ax.imshow(subplot_data.T, interpolation='none', cmap='gray')
ax.set_xticks([])
ax.set_yticks([])
#ax.invert_yaxis()

plt.tight_layout()
#fig = plt.figure()
#plt.imshow(subplot_data, interpolation='nearest')

plt.show()
