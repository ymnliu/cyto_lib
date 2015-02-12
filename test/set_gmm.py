import numpy as np

from pyclass import train_test
from plot import gen_set_gmm


def unique_rows(data):
    uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
    return uniq.view(data.dtype).reshape(-1, data.shape[1])


from test.sample_list import overlap_list

n_sample = 50
n_bins = 30 
l_bins = [5*i for i in range(1, 9)] 

non_overlap_list = [(0, i, 0) for i in range(n_sample)]


over_non_over = [non_overlap_list, overlap_list]

labels = [0, 1]
    
llogmat = []
lrspmat = []
lspot = []

for label in labels:
    s_list = over_non_over[label][:n_sample]

    for sid in s_list:
        log_mat, rsp_mat, spot = gen_set_gmm(sid, label)
        llogmat.append(log_mat)
        lrspmat.append(rsp_mat)
        lspot.append(spot)

for n_bins in l_bins:
    train = np.array([])
    train_truth = np.array([])
    
    for i in range(2 * n_sample):
        log_mat = llogmat[i]
        rsp_mat = lrspmat[i]
        spot = lspot[i]
        log_bins, lbin = np.histogram(log_mat, bins=n_bins, density=True)
        rsp_bins, rbin = np.histogram(rsp_mat, bins=n_bins, density=True)
        sf = spot.get_features()

        #bin_res = np.concatenate((log_bins, rsp_bins, sf))
        bin_res = np.concatenate((log_bins, lbin, rsp_bins, rbin, sf))
        #bin_res = np.concatenate((log_bins, lbin, rsp_bins, rbin))
        #bin_res = sf
        #print bin_res
        label = i / n_sample

        if train.shape[0] is 0:
            train = bin_res
            train_truth = np.array([label])
        else:
            train = np.vstack((train, bin_res))
            train_truth =  np.hstack((train_truth, label))

    print "#" * 80
    print "n_sample: %d\nn_bins: %d\n " % (n_sample, n_bins)
     
    for nfolds in [3, 6, 9]:
        print "NFold: %d" % (nfolds)
        train_test.classification_val(train, train_truth, nfolds)
