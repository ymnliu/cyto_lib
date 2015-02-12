import numpy as np
import matplotlib.pyplot as plt

from cyto.image import CytoImage
from cyto.util import load_cyto_list


image_list = np.array([[0, 1, 2], [0, 7, 8], [7, 10, 6]])
# image_list = np.array([[0, 1, 2],[0, 11, 8], [7, 10, 6]])

subplot_idx = 1

ax = []
for label in [0, 1, 2]:
    img_list = image_list[label]
    flist = load_cyto_list(label)
    for idx in img_list:
        # print flist[idx]
        img = CytoImage(flist[idx])
        ax.append(plt.subplot(3, 3, subplot_idx))
        plt.imshow(img.data, interpolation='none')
        ax[subplot_idx - 1].set_xticks([])
        ax[subplot_idx - 1].set_yticks([])

        tmpstr = "(" + str(subplot_idx) + ")"
        ax[subplot_idx - 1].set_xlabel(tmpstr)
        subplot_idx += 1

ax[0].set_ylabel("Monosomy")
ax[3].set_ylabel("Disomy")
ax[6].set_ylabel("Trisomy")

plt.tight_layout()
plt.show()
