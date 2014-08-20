from util import open_cyto_tiff as ot
from spot import CytoSpot

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, dilation, disk
from skimage.measure import regionprops, label

'''
def get_region_feature(img, region_list):
    result = []
    for region in region_list:
	f = []
	f.append(region.area)
	minr, minc, maxr, maxc = region.bbox
	
	mat = img[minr: maxr, minc: maxc]
	
	f.append(mat.sum())
	f.append( (mat**2).sum() )
	f.append(mat.sum() / (region.area * 1.))
	
	result.append(f)

    return result
'''

def get_region_spots(img, region_list):
    
    spot_list = []
    for region in region_list:
	minr, minc, maxr, maxc = region.bbox
	
	mat = img[minr: maxr, minc: maxc]
	spot_list.append(CytoSpot(mat))
    
    return spot_list

def get_mask(img_org):
    noise_ratio = 0.3
    img = img_org > noise_ratio * img_org.max()
    skeleton = skeletonize(img)
    
    mask = dilation(skeleton > 0, disk(2))
    return mask


def get_local_spots(img):
    
    mask = get_mask(img)
    masked_img = mask * img
    labeled_img = label(mask)

    regions = regionprops(labeled_img)
    
    return get_region_spots(img, regions)

def get_local_feature(img):
    '''
    mask = get_mask(img)
    masked_img = mask * img
    labeled_img = label(mask)

    regions = regionprops(labeled_img)
    
    result = get_region_feature(img, regions)
    '''
    
    return result

