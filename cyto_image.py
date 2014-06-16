import open_cyto_tiff as ot
import cyto_local_feature
from cyto_spot import CytoSpot

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, dilation, disk
from skimage.measure import regionprops, label

class CytoImage:
    'Class of cytometry image'
    data = np.asarray([])
    noise_ratio = .3

    def __init__(self, path):
	self.data = ot.open_cyto_tiff(path)

    def open_image(self, path):
	self.data = ot.open_cyto_tiff(path)
	return self.data

    def cyto_show(self):
	plt.imshow(self.data)
	plt.show()

    def get_mask(self):
	img = self.data > self.noise_ratio * self.data.max()
	skeleton = skeletonize(img)
	
	mask = dilation(skeleton > 0, disk(2))
	return mask

    def get_cyto_spots(self):
	img = self.data
	mask = self.get_mask()
	masked_img = mask * img
	labeled_img = label(mask)

	regions = regionprops(labeled_img)
    	
	spot_list = []
	for region in regions:
	    minr, minc, maxr, maxc = region.bbox
	    
	    mat = img[minr: maxr, minc: maxc]
	    spot_list.append(CytoSpot(mat))
	
	return spot_list
