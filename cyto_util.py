import os 
import sys
import open_cyto_tiff as ot

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

