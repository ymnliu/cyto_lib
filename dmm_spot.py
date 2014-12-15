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
import matplotlib.patches as mpatches
from matplotlib import cm
from sklearn import mixture

import  load_single_image as ls
from cyto.util import serialize
from cyto.feature import get_aic_bic_res

n_samples = 300
#label, idx = (2, 6)
label, idx = (1, 8)
label, idx = (2, 22)
label, idx = (2, 23)
#label, idx = (2, 33)
#label, idx = (2, 40)
#label, idx = (2, 45)
#label, idx = (2, 55)
#label, idx = (2, 85)
#label, idx = (1, 8)
#label, idx = (2, 500)
label, idx = (0, 2)

img = ls.load_single_image(label, idx)

img.show_each_spot()

imgdata = img.data
spots = img.spots

subplot_data = spots[0].data.T
#subplot_data = imgdata.T

ncomponents = 2 

ctype = ['spherical', 'tied', 'diag', 'full']

X_train = np.floor(serialize(subplot_data))

#np.savetxt("subplot_2_8.txt", X_train, fmt='%s')

#X_train = (X_train - np.mean(X_train))/ np.mean(X_train)

print "########################################"
print X_train

clf = mixture.DPGMM(n_components=ncomponents, covariance_type=ctype[1], n_iter=200, alpha=1., 
        verbose=True, random_state=20, params='m', init_params='wc', thresh=.00001)
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
#print clf.covars_.shape
#print clf.covars_
print clf.precs_
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
ax = fig.add_subplot(2, 2, 1)

#CS = ax.contour(X, Y, Z)
#CB = ax.colorbar(CS, shrink=0.8, extend='both')
#ax.scatter(X_train[:, 0], X_train[:, 1], .8)

truncate_ratio = 2
thres =  1 / float(truncate_ratio * ncomponents) #0.5 / ncomponents

#thres = 0
print "%d compoents larger than %.2f" % (len(np.argwhere(clf.weights_ > thres)), thres)

for cp_idx in range(0, ncomponents):
    if clf.weights_[cp_idx] > thres: 
        
        str_weight = "%.2f" % clf.weights_[cp_idx] 
	ax.scatter(clf.means_[cp_idx, 0].T, clf.means_[cp_idx, 1].T, c='r',marker='o')
	ax.text(clf.means_[cp_idx, 0].T, clf.means_[cp_idx, 1].T, str_weight, color='r' )

CS = ax.contour(X, Y, Z)
ax.imshow(subplot_data.T, interpolation='none',cmap='gray')

ax.set_xlabel("(a)")
ax.set_xticks([])
ax.set_yticks([])
#ax.invert_yaxis()

#ax = fig.add_subplot(1, 2, 2, aspect='equal')
ax = fig.add_subplot(2, 2, 2)
#ax.scatter(clf.means_[:, 0].T, clf.means_[:, 1].T, c='r',marker='o')
ax.imshow(img.data, interpolation='nearest')
#ax.set_title(str(clf.n_components))
ax.set_xlabel("(b)")
ax.set_xticks([])
ax.set_yticks([])

for spot in spots:
    spot = spots[0]
    minr = (spot.origin[0] - spot.epd_sz) 
    maxr = spot.origin[0]
    minc = (spot.origin[1] - spot.epd_sz)
    maxc = (spot.origin[1] )
    
    rect = mpatches.Rectangle((spot.origin[1], spot.origin[0]), 
	    spot.size[1], spot.size[0],
	    fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)
    
ax4 = fig.add_subplot(2, 2, 4)
aic, bic = get_aic_bic_res(spot.data)
x = np.arange(len(aic)) + 1
ax4.plot(x, aic, '-')
ax4.plot(x, bic, 'r-')

ax = fig.add_subplot(2,2,3, projection='3d')

surf = ax.plot_wireframe(X, Y, Z, cmap=cm.coolwarm, rstride=5, cstride=5)

ax = ax4

plt.tight_layout()
#fig = plt.figure()
#plt.imshow(subplot_data, interpolation='nearest')

plt.show()
