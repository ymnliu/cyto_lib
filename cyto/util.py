import os
import sys
from os import walk
import shutil

import skimage.io
import numpy as np
import cv2
import time, datetime

root = '/home/sijialiu/cyto/data'

img_dirname = ['20140508_CytoImage/monosomy_05082014',
               '20140508_CytoImage/disomy_05082014',
               '20140508_CytoImage/trisomy_05082014']

img_dir_list = [root + '/' + x for x in img_dirname]


def print_block(str_print):
    print ""
    print "##" * 30
    print "##    " + str_print
    print "##" * 30
    print ""


def try_image(img_dir, file_list, index):
    filename = file_list[index]
    img_path = img_dir + '/' + filename
    # print 'opening: ' + img_path
    img = open_cyto_tiff(img_path)

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

    if not os.path.exists(img_dir):
        sys.exit('Dir cannot found: ' + img_dir)

    print 'searching the directory: ' + img_dir

    for dir_path, dir_names, filenames in walk(img_dir):
        f = [fi for fi in filenames if fi.endswith('.tif')]
        break

    f.sort()

    if list_len is 0:
        list_len = len(f)

    output_name = root + '/' + str(label) + '.dat'
    fs = open(output_name, 'w')

    processed_count = 0
    skip_count = 0
    count = 0
    for index in range(0, list_len):
        img_path = img_dir + '/' + f[index]
        # print 'opening: ' + img_path
        im16_org = open_cyto_tiff(img_path)

        if im16_org is None:
            skip_count += 1
            continue
        else:
            processed_count += 1
            fs.write(img_path + os.linesep)

        processed_rate = processed_count / (list_len * 1.)
        skip_rate = skip_count / (list_len * 1.)

        sys.stdout.write("Processing: %.1f%%\t" % (100 * processed_rate))
        sys.stdout.write("Skip rate: %.1f%%\r" % (100 * skip_rate))
        sys.stdout.flush()
        count += 1

    fs.close()
    print "%d out of %d files are generated, %.1f%% skipped.\n" % \
          (processed_count, list_len, 100 * skip_rate)


def load_cyto_list(label):
    list_name = root + '/' + str(label) + '.dat'
    return [line.rstrip('\n') for line in open(list_name)]


def open_cyto_tiff(path):
    try:
        # im_m16 = np.matrix(im16)
        # im_array = np.asarray(im16[:,:])
        im16 = skimage.io.imread(path, plugin='freeimage')
        return im16
    except ValueError:
        # print "Value Error: skip image: " + path
        return None
    except IOError:
        # print "IO Error: skip image: " + path
        return None


def open_cyto_tiff_opencv(path):
    im16 = None
    # im_m16 = np.matrix(im16)
    # im_array = np.asarray(im16[:,:])
    # im16 = skimage.io.imread(path, plugin='freeimage')
    im16 = cv2.imread(path, 0)
    return im16


def serialize(_data):
    data = np.floor(_data / 200)
    res = np.array([])
    it = np.nditer(data, flags=['multi_index'])
    while not it.finished:
        (d, x, y) = (it[0], it.multi_index[0], it.multi_index[1])
        dtmp = np.tile([x, y], (int(d), 1))
        if res.shape[0] is 0:
            res = dtmp
        else:
            res = np.vstack((res, dtmp))

        it.iternext()

    return res


def show_progress(prefix_str, process, total):
    processed_rate = process / (1. * total)
    if process > total - 1:
        print prefix_str + " Complete!"
    else:
        display_str = prefix_str + ": %.1f%%\r"
        sys.stdout.write(display_str % (100 * processed_rate))
        sys.stdout.flush()


def gen_time_stamp(format_str='%Y%m%d%H%M-%S%f'):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime(format_str)
    return st
