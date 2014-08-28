import cyto.util as cu
import matplotlib.pyplot as plt
for i in 0, 1, 2:
    cu.gen_all_cyto_list(i)

flist = cu.load_cyto_list(0)

fname = flist[5]

img = cu.open_cyto_tiff_opencv(fname)
print type(img)
plt.imshow(img)

plt.show()
