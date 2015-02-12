import os
from os import mkdir
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def main(argv):
    f = []
    arg =  argv
    print arg, len(arg)

    display = False
    output_dir = 'output_subplot'
    
    fname = arg.rsplit('/', 1)[1]
    print "fname: " + fname
    img_path = arg

    if not os.path.exists(output_dir):
	mkdir(output_dir)
	print 'Directory Created: ' + output_dir

    print 'showing ' + img_path

    im_uint16 = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_UNCHANGED)     
    im_f32 = np.float32(im_uint16)  

    fig = plt.figure(figsize=plt.figaspect(2.))
    fig.suptitle('Original: ' + fname)
    ax = fig.add_subplot(2, 1, 1)

    plt.imshow(im_uint16)

    if False:
	plt.imshow(im_uint16)
	plt.show()

    #################################################
    #   display 3d
    ################################################

    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    ax = fig.add_subplot(2, 1, 2, projection='3d')

    X = np.arange(0, im_f32.shape[1])
    Y = np.arange(0, im_f32.shape[0])
    X, Y = np.meshgrid(X, Y)
    Z = im_f32

    #print Z.shape, Z.max()
    plt.title('3D surf: ' + fname)
    surf = ax.plot_surface(X, Y, Z, rstride=1, 
	    cstride=1, linewidth=0, cmap=cm.coolwarm)
    #surf = scatter(X, Y, Z)

    #ax.set_zlim(Z.max())
    #ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)

    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    #fig.colorbar(surf, shrink=0.5, aspect=5)

    #plt.show()

    output_path = output_dir +'/3d_' + \
	    fname.rsplit('.', 2)[0] + '.png'
    display = False
    plt.savefig(output_path)
    print 'Figure Saved: ' + output_path

if __name__ == "__main__":
    print "run directly"
    main(sys.argv[1:])
else:
    print "run as module"

