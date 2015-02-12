from os import walk
from datetime import datetime

import fun_3d


output_dir = 'output_subplot'

sample_size = 20

f = []
i = 0

total = 0.0

input_dir = '../2014.2.21-Disomy-control-images'
for (dirpath, dirnames, filenames) in walk(input_dir):
    f.extend(filenames[:sample_size])
    break

ts = datetime.now()

for fname in f:
    t1 = datetime.now()
    img_path = input_dir + '/' + fname
    print ''
    print 'processing #' + str(i) + ': ' + img_path 
    fun_3d.main(img_path)
    i += 1
    t2 = datetime.now()
    delta = t2 - t1
    total += delta.microseconds // 1000
    print 'Time: ' + str(delta.microseconds // 1000) + ' ms.'

te = datetime.now()
duration = te - ts

print ''
print 'All done!'
print 'Time: ' + str(total // 1000) + ' sec.'
print 'Time: ' + str(duration.seconds ) + ' sec.'
