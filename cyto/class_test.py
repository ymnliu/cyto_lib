#!/usr/bin/env python

from cyto_image import CytoImage

import numpy

path = "../data/disomy_05082014/5537_Ch3.ome.tif"

img = CytoImage(path)
#img.open_image(path)
print img.data
