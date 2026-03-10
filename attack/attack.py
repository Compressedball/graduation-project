import torch
import numpy as np
import scipy.sparse as sp
import utils
from module import GCN

def GCNSprase(adj, features, labels):
    def __init__(self, adj, features, labels):
        self.adj = adj
        self.features = features
        self.labels = labels
        self.module = GCN(
            in_features=features.shape[1],
            hidden=[32, 16],
            out_features=labels.shape[1]
        )