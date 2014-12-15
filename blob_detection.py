from matplotlib import pyplot as plt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray
import numpy as np

from load_single_image import load_single_image

#image = data.hubble_deep_field()[0:500, 0:500]
#image_gray = rgb2gray(image)

data, img = load_single_image(2,2)

image_gray = img.data / np.max(img.data.astype(float)) * 256.

image = image_gray

#blobs_log = blob_log(image_gray, max_sigma=5, num_sigma=10, threshold=.5)
blobs_log = blob_log(image_gray)
# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

#blobs_dog = blob_dog(image_gray, max_sigma=50, threshold=.5)
blobs_dog = blob_dog(image_gray)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_doh = blob_doh(image_gray)
#blobs_doh = blob_doh(image_gray, max_sigma=50, threshold=.05)

blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

for blobs, color, title in sequence:
    fig, ax = plt.subplots(1, 1)
    ax.set_title(title)
    ax.imshow(image, interpolation='nearest')
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax.add_patch(c)

plt.show()

