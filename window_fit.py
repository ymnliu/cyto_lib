import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from sklearn import mixture

import  load_single_image as ls
from cyto.util import serialize, print_block
from cyto.feature import get_aic_bic_res
import gaussian_window as gw 

n_samples = 300
#label, idx = (2, 6)
#label, idx = (1, 8)
#label, idx = (1, 8)
label, idx = (0, 400)

lsdata, img = ls.load_single_image(label, idx)

img.show_each_spot()

imgdata = img.data
spots = img.spots

subplot_data = spots[0].data.T

popt, pcov =  gw.gaussian_plot_2d(label, idx)



