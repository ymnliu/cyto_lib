import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

from cyto.cauchy_func import twod_cauthy


# Create x and y indices
w = 20
h = 20
x = np.linspace(0, 10, w)
y = np.linspace(0, 8, h)
x, y = np.meshgrid(x, y)

# create data
data = twod_cauthy((x, y), 36, 5, 5, 1, 64)
print data.shape
data = data.reshape(w, h)
# plot twoD_Gaussian data generated above
plt.figure()
plt.imshow(data)
plt.colorbar()



#################################################
#   display 3d
################################################
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(2, 1, 2, projection='3d')

X = np.arange(0, data.shape[1])
Y = np.arange(0, data.shape[0])
X, Y = np.meshgrid(X, Y)
Z = data

#print Z.shape, Z.max()
plt.title('3D surf of the image')
surf = ax.plot_surface(X, Y, Z, rstride=1,
                       cstride=1, linewidth=0, cmap=cm.coolwarm)

# # add some noise to the data and try to fit the data generated beforehand
# initial_guess = (36, 1.5, 1.2, 0.1, 64)
#
# data_noisy = data + 0.2*np.random.normal(size=data.shape)
#
# print data_noisy.shape
#
# popt, pcov = opt.curve_fit(twod_cauthy, (x, y), data_noisy, p0=initial_guess)
#
# data_fitted = twod_cauthy((x, y), *popt)
#
# fig, ax = plt.subplots(1, 1)
# ax.hold(True)
# ax.imshow(data_noisy.reshape(20, 20), cmap=plt.cm.jet, origin='bottom',
#     extent=(x.min(), x.max(), y.min(), y.max()))
# ax.contour(x, y, data_fitted.reshape(20, 20), 8, colors='w')

plt.show()
