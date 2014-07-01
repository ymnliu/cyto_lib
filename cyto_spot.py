import numpy as np
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt

class CytoSpot:
    'class of spots in cytometry image'
    org_data = []
    data = []
    norm_data = []
    epd_sz = 1
    region = None
    origin = (-1, -1)
    size = ( -1, -1)
    bbox = (-1, -1, -1, -1)
    noise_ratio = .3
    
    def __init__(self, data):
	org_data = data
	(x, y) =  org_data.shape

	self.data = np.zeros((x + 2 * self.epd_sz, 
	    y + 2 * self.epd_sz))
	
	self.size = (x + 2 * self.epd_sz, y + 2 * self.epd_sz)
	self.data[self.epd_sz:x + self.epd_sz, 
		self.epd_sz:y + self.epd_sz] = org_data
	
	self.norm_data = self.data / self.data.sum()
	#print self.data

	mask = self.data > self.noise_ratio * self.data.max()

	labeled_img = label(mask)
		
	regions = regionprops(labeled_img)
	self.region = regions[0]

    def get_features(self):
	region = self.region
	result = []
	
	f = []
	minr, minc, maxr, maxc = region.bbox
	
	mat = self.data[minr: maxr, minc: maxc]
	width = maxc - minc
	height = maxr - minr

	f.append(region.area)
	f.append(region.extent )    #height width ratio
	f.append(region.convex_area)
	f.append(region.eccentricity)
	f.append(region.equivalent_diameter)
	
	
	f.append(region.solidity)   #Ratio of pixels in the region 
				    #to pixels of the convex hull image.

	f.append(mat.sum())
	f.append((mat**2).sum() / (region.area * 1.))
	f.append(mat.sum() / (region.area * 1.))
	
	result.append(f)

	return result

    def show_spot(self, ax):
	ax.plot(self.data)
	return ax 
	
