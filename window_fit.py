import  load_single_image as ls
from plot import gaussian_window as gw

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



