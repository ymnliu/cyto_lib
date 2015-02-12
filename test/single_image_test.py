import numpy as np

import matplotlib.pyplot as plt
from skimage.filter import gabor_kernel
from skimage import data

from load_single_image import load_single_image



############################################################################
##      function def
############################################################################





######################################################################
########        Begin script
######################################################################


data, img = load_single_image(2,2)

fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(3, 2, 1)

ax.imshow(img.data, interpolation='nearest')

ax = fig.add_subplot(3,2, 2)

data = img.data / np.max(img.data.astype(float)) * 256.

ax.imshow(data, interpolation='nearest')

kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)





plt.show()

