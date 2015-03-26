import cyto.helper as ch
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
import matplotlib.patches as mpatches
import os
import cyto.util
import numpy as np
from scipy.stats import beta


sample_size = 3000

path_prefix = "../../result"

spot_count_list = []

spot_max = np.zeros((3, sample_size, 5))

label_list = [2]

for label in label_list:
    print "\nLabel %d" % label

    fig_dir = "{}/{}".format(path_prefix, label)

    try:
        os.makedirs(fig_dir)
    except OSError:
        if not os.path.isdir(fig_dir):
            raise

    for i in range(sample_size):
        img = ch.load_single_image(label, i)

        spots = img.get_cyto_spots()
        spot_count_list.append(len(spots))

        for spot_id, spot in enumerate(spots):
            spot_max[label][i][spot_id] = np.max(spot.data)
            # fig_path = "{}/{:04}.png".format(fig_dir, i)
            # print fig_path
            # ax = plt.subplot(111)
            # ch.cyto_show_spots(img, ax)
            # plt.savefig(fig_path)
            # plt.close()

    # count occurrences
    print "Label, Count -> Percentage"
    for i in range(1, 5):
        print "%d, %d  -> %.3f" % \
              (label, i,
               spot_count_list.count(i) / float(len(spot_count_list)))

smax1 = spot_max[2]
smax1_nz = smax1[:, 1] > 0
sm_sorted = np.sort(smax1[smax1_nz], axis=1)
a = sm_sorted[:, -1] / sm_sorted[:, -2]
a = 1. / a

beta_a, beta_b, floc, fscale = beta.fit(a)

print beta_a, beta_b


# rv = beta(beta_a, beta_b)
rv = beta(beta_a, beta_b, loc=floc, scale=fscale)
x = np.linspace(beta.ppf(0.01, beta_a, beta_b, loc=floc, scale=fscale),
                beta.ppf(0.99, beta_a, beta_b, loc=floc, scale=fscale), 100)

fig_count = 2
fig, axarr = plt.subplots(2, 2)
fig.suptitle("Trisomy - bin # wrt spot max ratio\n%s\n num of cases: %d"
             % (__file__, a.shape[0]))


# ax.plot(a, linewidth=3, alpha=0.5, label='disomy max ratio')
for sub_id, bin_num in enumerate([10, 20, 40, 80]):
    sub_row, sub_col = sub_id / 2, sub_id % 2
    ax = axarr[sub_row][sub_col]

    ax.hist(a, bin_num, fc='b',  # histtype='stepfilled',
            alpha=0.3, normed=True)
    ax.set_title('bin_num: %d' % bin_num)
    ax.set_xlim(.0, 1.1)
    ax.plot(x, rv.pdf(x), 'k-', lw=2)
    ax.set_ylim([0, 2.5])
    ax.text(.5, 2.0, "beta param:\n a, b, loc, scale\n %.2f, %.2f, %.2f, %.2f" %
            (beta_a, beta_b, floc, fscale))
# plt.show()


real_trisomy_idx = smax1[:, 2] > 0

tri_sorted = np.array(np.sort(smax1[real_trisomy_idx], axis=1))

print len(tri_sorted)

tri_sum = np.sum(tri_sorted, axis=1)
tri_normed = (np.float_(tri_sorted).T / tri_sum)[2:5]

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tri_normed[0], tri_normed[1], tri_normed[2], c='r', marker='o')
plt.show()