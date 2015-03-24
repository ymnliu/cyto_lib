import cyto.helper as ch
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
import matplotlib.patches as mpatches
import os
import cyto.util
import numpy as np


sample_size = 3000

path_prefix = "../../result"

spot_count_list = []

spot_max = np.zeros((3, sample_size, 5))

label_list = [1]

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

smax1 = spot_max[1]
smax1_nz = smax1[:, 1] > 0
sm_sorted = np.sort(smax1[smax1_nz], axis=1)
a = sm_sorted[:, -1] / sm_sorted[:, -2]
a = 1. / a
print a.shape[0]
fig_count = 2
fig, axarr = plt.subplots(2, 2)
fig.suptitle("disomy - bin # wrt spot max ratio\n%s\n num of cases: %d" % (__file__, a.shape[0]))
# ax.plot(a, linewidth=3, alpha=0.5, label='disomy max ratio')
for sub_id, bin_num in enumerate([10, 20, 40, 80]):
    sub_row, sub_col = sub_id / 2, sub_id % 2
    axarr[sub_row][sub_col].hist(a, bin_num, fc='gray',
                                 histtype='stepfilled', alpha=0.3, normed=True)
    axarr[sub_row][sub_col].set_title('bin_num: %d' % bin_num)
    axarr[sub_row][sub_col].set_xlim(.0, 1.1)
plt.show()