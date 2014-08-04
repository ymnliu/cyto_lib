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

import  load_single_image as ls

n_samples = 300

lsdata = ls.data
imgdata = ls.img.data
# generate random sample, two components
np.random.seed(0)
C = np.array([[0., -0.7], [3.5, .7]])
X_train_decimal = np.r_[np.dot(np.random.randn(n_samples, 2), C),
                np.random.randn(n_samples, 2) + np.array([20, 20])]
X_train = np.ceil(lsdata)
#clf = mixture.DPGMM(n_components=3, covariance_type='full')
clf = mixture.DPGMM(n_components=5, covariance_type='diag', n_iter=100)
clf.fit(X_train)

print clf.weights_
print clf.means_
print clf.get_params()
x = np.linspace(0.0, imgdata.shape[0])
y = np.linspace(0.0, imgdata.shape[1])
X, Y = np.meshgrid(x, y)
XX = np.c_[X.ravel(), Y.ravel()]
#Z = np.log(-clf.score_samples(XX)[0])
Z = np.log(-clf.score_samples(XX)[0])
Z = Z.reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, aspect='equal')

#CS = ax.contour(X, Y, Z)
CS = ax.contour(X, Y, Z)
#CB = ax.colorbar(CS, shrink=0.8, extend='both')
ax.scatter(X_train[:, 0], X_train[:, 1], .8)
ax.scatter(clf.means_[:, 0].T, clf.means_[:, 1].T, c='r',marker='o')
ax.set_xlabel("(a)")
ax.set_xticks([])
ax.set_yticks([])
ax.invert_yaxis()

ax = fig.add_subplot(1, 2, 2, aspect='equal')
ax.scatter(clf.means_[:, 0].T, clf.means_[:, 1].T, c='r',marker='o')
ax.imshow(ls.img.data.T, interpolation='nearest')
#ax.set_title(str(clf.n_components))
ax.set_xlabel("(b)")
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.show()
