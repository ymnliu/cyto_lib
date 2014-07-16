import cyto_util as cu
import cyto_local_feature
import cyto_feature as cf
from cyto_spot import CytoSpot

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.morphology import skeletonize, dilation, disk
from skimage.measure import regionprops, label

class CytoImage:
    'Class of cytometry image'
    data = np.asarray([])
    noise_ratio = .3

    def __init__(self, path):
	self.data = cu.open_cyto_tiff(path)
	self.path = path
	self.spots = []
	#print "spots count init"

    def open_image(self, path):
	self.data = cu.open_cyto_tiff(path)
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
    	
	for region in regions:
	    minr, minc, maxr, maxc = region.bbox
	    
	    mat = img[minr: maxr, minc: maxc]
	    spot = CytoSpot(mat)
	    spot.origin = (minr - spot.epd_sz, minc - spot.epd_sz)
	    self.spots.append(spot)
	
	return self.spots

    def cyto_show_spots(self, ax=None):
	if ax is None:
	    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))

	ax.imshow(self.data)

	for spot in self.spots:
	    minr = (spot.origin[0] - spot.epd_sz) 
	    maxr = spot.origin[0]
	    minc = (spot.origin[1] - spot.epd_sz)
	    maxc = (spot.origin[1] )
	    
	    rect = mpatches.Rectangle((spot.origin[1], spot.origin[0]), 
		    spot.size[1], spot.size[0],
		    fill=False, edgecolor='red', linewidth=1)
	    ax.add_patch(rect)
	    
	"""
	Show how to modify the coordinate formatter to 
	report the image "z"
	value of the nearest pixel given x and y
	"""
	
	X = self.data
	numrows, numcols = X.shape
	def format_coord(x, y):
	    col = int(x+0.5)
	    row = int(y+0.5)
	    if col>=0 and col<numcols and row>=0 and row<numrows:
		z = X[row,col]
		return 'x=%1.2f, y=%1.2f, z=%1.2f'%(x, y, z)
	    else:
		return 'x=%1.2f, y=%1.2f'%(x, y)
	
	ax.format_coord = format_coord	
	#plt.show()
	return ax

    def show_each_spot(self):
	fig = plt.figure(figsize=plt.figaspect(1.))
	idx = 1
	for spot in self.spots:
	    ax = fig.add_subplot(2,2, idx)
	    idx += 1
    	    #ax.plot(spot.data)#show spot in curve. can be useful

	    ax.imshow(spot.data, interpolation='nearest')
	
	ax = fig.add_subplot(2,2, idx)
	ax.imshow(self.data, interpolation='nearest')
	
	for spot in self.spots:
	    minr = (spot.origin[0] - spot.epd_sz) 
	    maxr = spot.origin[0]
	    minc = (spot.origin[1] - spot.epd_sz)
	    maxc = (spot.origin[1] )
	    
	    rect = mpatches.Rectangle((spot.origin[1], spot.origin[0]), 
		    spot.size[1], spot.size[0],
		    fill=False, edgecolor='red', linewidth=1)
	    ax.add_patch(rect)
	#plt.show()
	return fig

    def get_feature(self):
	return cf.get_image_feature(self.data)
