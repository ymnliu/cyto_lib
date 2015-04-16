import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def twod_cauthy((x, y), *p):
    """ 2d cauthy function with input of (x, y), p and offset"""
    (amplitude, xo, yo, b, offset) = p
    xo = float(xo) + .5
    yo = float(yo) + .5
    b = float(b)

    g = offset + amplitude / (1. * (b + (x - xo) * (y - yo)))
    return g.ravel()


def cauthy_fit_2d(data2d):
    data = data2d

    (w, h) = data.shape
    x = range(0, h)
    y = range(0, w)
    x, y = np.meshgrid(x, y)

    init_0 = (data.max(), h / 2, w / 2, 1, 0)

    popt, pcov = curve_fit(twod_cauthy, (x, y), data.ravel(), p0=init_0)
    perr = np.sum(np.sqrt(np.diag(pcov)))
    # perr = 0
    try:
        if np.isinf(perr) or np.isnan(perr) or perr > 1000:
            if perr > 1000:
                raise Exception('large')
            if np.isinf(perr):
                raise Exception('inf')
            if np.isnan(perr):
                raise Exception('nan')
        return np.append(popt, perr)
    except (RuntimeError, Exception) as exception:
        if isinstance(exception, RuntimeError):
            exp_str = "opt404"
        else:
            exp_str = exception.args[0]
        print exp_str

        return exp_str

