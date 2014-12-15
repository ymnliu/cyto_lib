import cyto.util as cu
import matplotlib.pyplot as plt

import os

#for i in 0, 1, 2:
#    cu.gen_all_cyto_list(i)

root = '../data_view_eps'


def prepare_dir(label):
    directory = root + '/' + str(label)
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory

sample_num  = 1000

for i in 0,1,2:
    flist = cu.load_cyto_list(i)
    directory = prepare_dir(i)

    for idx  in range(sample_num):
        fname = flist[idx]
        img = cu.open_cyto_tiff(fname)
        #path = directory + '/' + str(idx) + '_' + fname.split('/')[-1] + '.png'
        path = directory + '/' + str(idx) + '_' + fname.split('/')[-1] + '.eps'

        #plt.imshow(img, interpolation='nearest')

        #plt.imsave(path, img, format='png')
        plt.imsave(path, img, format='eps', cmap=plt.get_cmap('gray'))
    
#plt.show()
