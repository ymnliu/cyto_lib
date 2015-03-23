"""
=============================================
Density Estimation for a mixture of Gaussians
=============================================

Plot the density estimation of a mixture of two gaussians. Data is
generated from two gaussians with different centers and covariance
matrices.
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture

import load_single_image as ls
from cyto.util import serialize

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

n_samples = 3
label = 0
idx = 5

index_list = []
res_list = []

for label in range(0, 3):
    for idx in range(0, 10):

        try:
            lsdata, img = ls.load_single_image(label, idx)
        except:
            print label, idx
            print "skiped. "
            continue

        imgdata = img.data
        spots = img.spots

        # subplot_data = spots[0].data.T
        subplot_data = imgdata

        ctype = ['spherical', 'tied', 'diag', 'full']

        aic_res = np.zeros(5)
        bic_res = np.zeros(5)

        res_idx = 0

        for ncomponents in range(1, 6):
            X_train = np.ceil(serialize(subplot_data))
            # clf = mixture.DPGMM(n_components=3, covariance_type='full')
            clf = mixture.GMM(n_components=ncomponents, covariance_type=ctype[1],
                              n_iter=100)
            clf.fit(X_train)

            aic_res[res_idx] = clf.aic(X_train)
            bic_res[res_idx] = clf.bic(X_train)
            # print clf.weights_
            # print clf.means_
            #print clf.get_params()
            x = np.linspace(0.0, subplot_data.shape[0])
            y = np.linspace(0.0, subplot_data.shape[1])
            X, Y = np.meshgrid(x, y)
            XX = np.c_[X.ravel(), Y.ravel()]
            #Z = np.log(-clf.score_samples(XX)[0])
            Z = np.log(-clf.score_samples(XX)[0])
            Z = Z.reshape(X.shape)

            res_idx += 1

        print "#" * 70
        print label, idx
        print aic_res, bic_res

        print "argmin:"
        print np.argmin(aic_res) + 1, np.argmin(bic_res) + 1
        print "num components"
        print img.get_n_component()
        res_list.append((label + 1, idx, np.argmin(aic_res) + 1,
                         np.argmin(bic_res) + 1, img.get_n_component()))

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, aspect='equal')
        ax.imshow(subplot_data)

print '\n'.join(map(str, res_list))
# plt.show()


'''
#CS = ax.contour(X, Y, Z)
CS = ax.contour(X, Y, Z)
#CB = ax.colorbar(CS, shrink=0.8, extend='both')
#ax.scatter(X_train[:, 0], X_train[:, 1], .8)

thres =  0 #0.5 / ncomponents

for cp_idx in range(0, ncomponents):
    if clf.weights_[cp_idx] > thres: 
	ax.scatter(clf.means_[cp_idx, 0].T, clf.means_[cp_idx, 1].T, c='r',marker='o')

ax.imshow(subplot_data.T, interpolation='none',cmap='gray')

ax.set_xlabel("(a)")
ax.set_xticks([])
ax.set_yticks([])
#ax.invert_yaxis()

ax = fig.add_subplot(1, 2, 2, aspect='equal')
#ax.scatter(clf.means_[:, 0].T, clf.means_[:, 1].T, c='r',marker='o')
ax.imshow(img.data, interpolation='nearest')
#ax.set_title(str(clf.n_components))
ax.set_xlabel("(b)")
ax.set_xticks([])
ax.set_yticks([])

for spot in spots:
    minr = (spot.origin[0] - spot.expand_size)
    maxr = spot.origin[0]
    minc = (spot.origin[1] - spot.expand_size)
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

fig = plt.figure(figsize=plt.figaspect(1.))
idx = 1

for spot in spots:
    ax = fig.add_subplot(2,2, idx)
    idx += 1
    #ax.plot(spot.data)#show spot in curve. can be useful

    ax.imshow(spot.data, interpolation='nearest')

ax = fig.add_subplot(2,2, idx)
ax.imshow(imgdata, interpolation='nearest')

for spot in spots:
    minr = (spot.origin[0] - spot.expand_size)
    maxr = spot.origin[0]
    minc = (spot.origin[1] - spot.expand_size)
    maxc = (spot.origin[1] )
    
    rect = mpatches.Rectangle((spot.origin[1], spot.origin[0]), 
	    spot.size[1], spot.size[0],
	    fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)


#plt.show()

'''
