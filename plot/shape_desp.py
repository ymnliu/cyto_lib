import cyto.helper as ch
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
import matplotlib.patches as mpatches


sample_size = 1000

spot_count_list = []
for label in range(3):
    print "\nProcessing Label %d" % label
    print "Label, Count -> Percentage"
    for i in range(sample_size):
        img = ch.load_single_image(label, i)
        spot_count_list.append(len(img.get_cyto_spots()))

    # count occurrences
    for i in range(1, 5):
        print "%d, %d  -> %.3f" % \
              (label, i,
               spot_count_list.count(i) / float(len(spot_count_list)))

# ch.show_each_spot(img)

