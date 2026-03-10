import numpy as np
import scipy.sparse as sp
import random

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

def normalize_sparse_coo_adj(adj):
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_features(features):
    row_sum = features.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    features = features / row_sum
    return features

def split_data(labels, split_plan):
    idx = np.arange(labels.shape[0])
    classes = np.unique(labels)

    if split_plan == 1:
        train_per_class = 20
        val_size = 500
        test_size = 1000

        idx_train = []
        for c in classes:
            idx_c = np.where(labels == c)[0]
            np.random.shuffle(idx_c)
            idx_train.extend(idx_c[:train_per_class])
        idx_train = np.array(idx_train)
        remaining = np.setdiff1d(idx, idx_train)
        np.random.shuffle(remaining)
        idx_val = remaining[:val_size]
        idx_test = remaining[val_size:(val_size + test_size)]

    if split_plan == 2:
        train_ratio = 0.1
        val_ratio = 0.1
        test_ratio = 0.8
        
        idx_train = []
        idx_val = []
        idx_test = []

        for c in classes:
            idx_c = np.where(labels == c)[0]
            np.random.shuffle(idx_c)

            n_total = len(idx_c)
            n_train = int(train_ratio * n_total)
            n_val = int(val_ratio * n_total)
            n_test = n_total - n_train - n_val

            idx_train.extend(idx_c[:n_train])
            idx_val.extend(idx_c[n_train:n_train+n_val])
            idx_test.extend(idx_c[n_train+n_val:])

        idx_train = np.array(idx_train)
        idx_val = np.array(idx_val)
        idx_test = np.array(idx_test)

        np.random.shuffle(idx_train)
        np.random.shuffle(idx_val)
        np.random.shuffle(idx_test)

    # for c in classes:
    #     a = np.where(labels[idx] == c)[0]
    #     print(a.shape)
    # print()
    # for c in classes:
    #     a = np.where(labels[idx_train] == c)[0]
    #     print(a.shape)
    # print()
    # for c in classes:
    #     a = np.where(labels[idx_val] == c)[0]
    #     print(a.shape)
    # print()
    # for c in classes:
    #     a = np.where(labels[idx_test] == c)[0]
    #     print(a.shape)
    # print()

    return idx_train, idx_val, idx_test