#from load_image import spots, img
from cyto_image import CytoImage
from cyto_util import load_cyto_list

import numpy as np

from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define model function to be used to fit to the data above:
def gaussian_function(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def twoD_Gaussian_theta((x, y), *p):
    ' 2d gaussian function with input of theta, (x, y) and offset'
    (amplitude, xo, yo,
		sigma_x, sigma_y, theta, offset) = p
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) +  (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) +  (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

def twoD_Gaussian_cov((x, y), *p):
    
    (A, mu, cov_matrix, offset) = p
    xo = mu[0]
    yo = mu[1]

    a = cov_matrix[0][0]
    b = cov_matrix[0][1]
    c = cov_matrix[1][1]
    g = offset + A * np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

def gaussian_fit_1d():
    fig = plt.figure()
    idx = 0
    for spot in spots:
	ax = fig.add_subplot(2, 2, idx + 1)
	idx += 1
	#spot = spots[1]
	(w, h) = spot.data.shape
	print w, h
	data = spot.data[w/2]

	win_sz = len(data)
	mu = win_sz / 2

	std = 1

	window = signal.gaussian(win_sz, std=std) * data.max()
	ax.plot(window)
	plt.title(r"Gaussian window ($\sigma$=" + str(std) + ")" )
	plt.ylabel("Amplitude")
	plt.xlabel("Sample No." + str(idx))
	p0 = [1., 0., 1.]
	x_e = range(win_sz) 
	coeff, var_matrix = curve_fit(gaussian_function, 
		x_e , data, p0=p0)
	
	print coeff
	x = np.linspace( 0, win_sz, 100)
	y = gaussian_function(x, *coeff)

	plt.plot(x, y, 'g-', label='fd', linewidth=5, alpha=0.4)

	plt.plot(data, 'r--')

    ax = fig.add_subplot(2,2, idx + 1)
    ax.imshow(img.data)
    plt.show()

def gaussian_fit_2d(data2d):
    data = data2d
    
    (w, h) = data.shape
    x = range(0, h)
    y = range(0, w)
    x, y = np.meshgrid(x, y)
    
    init_0 = (data.max(), h/2, w/2, 1, 1, 0, 0)	

    popt, pcov = curve_fit( twoD_Gaussian_theta, (x, y), data.ravel(), p0=init_0)
    return  popt 
	

def gaussian_plot_2d():
    
    label = 2
    img_idx = 104
    img = CytoImage(load_cyto_list(label)[img_idx])
    spots = img.get_cyto_spots() 

    fig = plt.figure()
    idx = 0
    
    for spot in spots:
	ax = fig.add_subplot(2, 2, idx + 1)
	idx += 1
	#spot = spots[1]
	(w, h) = spot.data.shape
	data = spot.data
	
	#x = np.linspace(0, h-1, h)
	#y = np.linspace(0, w-1, w)
	x = range(0, h)
	y = range(0, w)
	x, y = np.meshgrid(x, y)
	
	#print (x, y)
	#cov0 = np.eye(2)
	#mu0 = np.array([h/2, w/2])
	#init_0 = (data.max(), mu0 , cov0)	
	init_0 = (data.max(), h/2, w/2, 1, 1, 0, 0)	

	popt, pcov = curve_fit( twoD_Gaussian_theta, (x, y), data.ravel(), p0=init_0)
	print  popt 
	#print pcov
	data_fitted = twoD_Gaussian_theta((x, y), *popt)
	
#	popt, pcov = curve_fit( twoD_Gaussian_cov, (x, y), data.ravel(), p0=init_0)
#	data_fitted = twoD_Gaussian_cov((x, y), *popt)
	
	ax.imshow(data.reshape(w,h))
	ax.contour(x, y, data_fitted.reshape(w, h))

	#data = twoD_Gaussian((x, y), 3, 100, 100, 20, 40, 0, 10)

    ax = fig.add_subplot(2, 2, idx + 1)
    ax.imshow(img.data)
    plt.show()

gaussian_plot_2d()
