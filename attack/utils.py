import numpy as np
import scipy.sparse as sp

def load_data(name):
    names = ["cora", "citeseer", "pubmed"]
    assert name in names, "Dataset not found. Available datasets: " + str(names)
    path_adj = "../data/" + name + "/"+ name + "_adj.npz"
    adj = sp.load_npz(path_adj)
    path_features = "../data/" + name + "/"+ name + "_features.npy"
    features = np.load(path_features)
    path_labels = "../data/" + name + "/"+ name + "_labels.npy"
    labels = np.load(path_labels)
    return adj, features, labels