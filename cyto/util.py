import os 
import sys
import skimage.io

import numpy as np

from os import listdir, mkdir, walk

img_dir_list =[ '../data/monosomy_05082014',
		'../data/disomy_05082014',
		'../data/trisomy_05082014']

def print_block(str_print):
    print ""
    print "##" * 30
    print "##    " + str_print
    print "##" * 30
    print ""

def try_image(img_dir, file_list, index):
    filename = file_list[index]
    img_path = img_dir + '/' + filename
    #print 'opening: ' + img_path
    img = ot.open_cyto_tiff(img_path)

    if img is None:
	print "skip image: " + img_path 
	return try_image(img_dir, file_list, index + 1)

    print 'opening: ' + img_path
    return img, img_path 

def remove_all(dir_path):
    if os.path.exists(dir_path):
	shutil.rmtree(dir_path)		
	print "Directory files removed: " + dir_path
    else:
	print "Directory: \'" + dir_path + "\' does not exist"

def gen_all_cyto_list(label, list_len=0):

    img_dir = img_dir_list[label] 
    f = []
    
    print "reading " + img_dir + " ..."

    if not os.path.exists(img_dir):
	sys.exit( 'Dir cannot found: ' + img_dir )

    print 'searching the directory: ' + img_dir  

    for (dirpath, dirnames, filenames) in walk(img_dir):
	f.extend(filenames)
	break
    
    f.sort()

    if list_len is 0:
	list_len = len(f)
    
    output_name = '../data/' + str(label) + '.dat' 
    fs = open(output_name, 'w')
    
    processed_count = 0
    skip_count = 0

    for index in range(0, list_len):
	img_path = img_dir + '/' + f[index]
	#print 'opening: ' + img_path
	im16_org = ot.open_cyto_tiff(img_path)
	
	if im16_org is None:
	    skip_count +=  1
	    continue
	else:
	    processed_count +=  1
	    fs.write(img_path + os.linesep)
	
	sys.stdout.write("Processing: %d%%  \t " % 
		(100 * processed_count / (list_len - skip_count) + 1) )
	sys.stdout.write("skip rate: %d%%   \r" % 
		(skip_count / (list_len * 1.)) )
	sys.stdout.flush()
    
    fs.close()
    
def load_cyto_list(label):
    list_name = '../data/' + str(label) + '.dat' 
    return [line.rstrip('\n') for line in open(list_name)]

def open_cyto_tiff(path):
    try:
	#im16 = cv2.cv.LoadImage(path, cv2.CV_LOAD_IMAGE_UNCHANGED)   
	#im_m16 = np.matrix(im16)
	#im_array = np.asarray(im16[:,:])
	im16 = skimage.io.imread(path, plugin='freeimage')
	return im16
    except:
	#print "skip image: " + path
	return None

def serialize(_data):
    data = np.floor(_data/100) 
    res = np.array([])	
    it = np.nditer(data, flags=['multi_index'])
    while not it.finished:
	(d, x, y) =  (it[0], it.multi_index[0], it.multi_index[1])
	dtmp = np.tile([x, y], (d, 1))
	if res.shape[0]is 0:
	    res = dtmp
	else:
	    res = np.vstack((res, dtmp))

	it.iternext()
    
    return res

