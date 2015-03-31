import numpy as np
from scipy.optimize import curve_fit


# Define model function to be used to fit to the data above:
def gaussian_function(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


def twod_gaussian_theta((x, y), *p):
    """ 2d gaussian function with input of theta, (x, y) and offset"""
    (amplitude, xo, yo, sigma_x, sigma_y, theta, offset) = p
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) +  (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) +  (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()


def twod_gaussian_cov((x, y), *p):
    ' 2d gaussian function with input of covariance matrix, (x, y) and offset'
    (A, mu, cov_matrix, offset) = p
    xo = mu[0]
    yo = mu[1]

    a = cov_matrix[0][0]
    b = cov_matrix[0][1]
    c = cov_matrix[1][1]
    g = offset + A * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))
    return g.ravel()


def gaussian_fit_2d(data2d):
    data = data2d
    
    (w, h) = data.shape
    x = range(0, h)
    y = range(0, w)
    x, y = np.meshgrid(x, y)
    
    init_0 = (data.max(), h/2, w/2, 1, 1, 0, 0)	
    try:
        popt, pcov = curve_fit(twod_gaussian_theta, (x, y), data.ravel(), p0=init_0)
        perr = np.sum(np.sqrt(np.diag(pcov)))
        if np.isinf(perr) or np.isnan(perr) or perr > 1000:
            raise Exception('perr is inf or nan')
        return np.append(popt, perr)
    except (RuntimeError, Exception) as exception:
        print exception
        return None

