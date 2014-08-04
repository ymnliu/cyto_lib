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
from sklearn import mixture

import  load_single_image as ls
from cyto_util import serialize

n_samples = 300
label = 1
idx = 8

lsdata, img = ls.load_single_image(label, idx)

imgdata = img.data
spots = img.spots

subplot_data = spots[0].data.T

ncomponents = 2 

ctype = ['spherical', 'tied', 'diag', 'full']

X_train = np.ceil(serialize(subplot_data))
#clf = mixture.DPGMM(n_components=3, covariance_type='full')
clf = mixture.GMM(n_components=ncomponents, covariance_type=ctype[0],
	n_iter=100)
clf.fit(X_train)

print clf.aic(X_train)
print clf.weights_
print clf.means_
print clf.get_params()
x = np.linspace(0.0, subplot_data.shape[0])
y = np.linspace(0.0, subplot_data.shape[1])
X, Y = np.meshgrid(x, y)
XX = np.c_[X.ravel(), Y.ravel()]
#Z = np.log(-clf.score_samples(XX)[0])
Z = np.log(-clf.score_samples(XX)[0])
Z = Z.reshape(X.shape)

fig = plt.figure()
#ax = fig.add_subplot(1, 2, 1, aspect='equal')
ax = fig.add_subplot(1, 2, 1)

#CS = ax.contour(X, Y, Z)
#CB = ax.colorbar(CS, shrink=0.8, extend='both')
#ax.scatter(X_train[:, 0], X_train[:, 1], .8)

thres =  0 #0.5 / ncomponents

for cp_idx in range(0, ncomponents):
    if clf.weights_[cp_idx] > thres: 
	ax.scatter(clf.means_[cp_idx, 0].T, clf.means_[cp_idx, 1].T, c='r',marker='o')

CS = ax.contour(X, Y, Z)
ax.imshow(subplot_data.T, interpolation='none',cmap='gray')

ax.set_xlabel("(a)")
ax.set_xticks([])
ax.set_yticks([])
#ax.invert_yaxis()

#ax = fig.add_subplot(1, 2, 2, aspect='equal')
ax = fig.add_subplot(1, 2, 2)
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

plt.tight_layout()
plt.show()

############################################################
##  second plot
############################################################
'''
fig = plt.figure()
idx = 1

for spot in spots:
    ax = fig.add_subplot(2,2, idx)
    idx += 1
    #ax.plot(spot.data)#show spot in curve. can be useful

    ax.imshow(spot.data, interpolation='nearest')

ax = fig.add_subplot(2,2, idx)
ax.imshow(imgdata, interpolation='nearest')

for spot in spots:
    minr = (spot.origin[0] - spot.epd_sz) 
    maxr = spot.origin[0]
    minc = (spot.origin[1] - spot.epd_sz)
    maxc = (spot.origin[1] )
    
    rect = mpatches.Rectangle((spot.origin[1], spot.origin[0]), 
	    spot.size[1], spot.size[0],
	    fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)

#plt.show()
'''
