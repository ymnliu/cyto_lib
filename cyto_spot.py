import numpy as np
from skimage.measure import regionprops, label

class CytoSpot:
    'class of spots in cytometry image'
    org_data = []
    data = []
    epd_sz = 1
    region = None

    def __init__(self, data):
	org_data = data
	(x, y) =  org_data.shape
	self.data = np.zeros((x + 2 * self.epd_sz, 
	    y + 2 * self.epd_sz))

	self.data[self.epd_sz:x + self.epd_sz, 
		self.epd_sz:y + self.epd_sz] = org_data
	
	mask = self.data > 0

	labeled_img = label(mask)
		
	regions = regionprops(labeled_img)
	self.region = regions[0]

	
    def get_spot_features(self):
	region = self.region
	result = []
	
	f = []
	f.append(region.area)
	minr, minc, maxr, maxc = region.bbox
	
	mat = img[minr: maxr, minc: maxc]
	
	f.append(mat.sum())
	f.append( (mat**2).sum() )
	f.append(mat.sum() / (region.area * 1.))
	
	result.append(f)

	return result
