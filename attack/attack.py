import torch
import numpy as np
import scipy.sparse as sp
import utils
import module
from module import GCN, GAT, GraphSAGE
from torch.nn import functional as F

class GCNSprase():
    def __init__(self, adj, features, labels,
                 idx_train, idx_val, idx_test,
                 device,
                 dropout=0.5, lr=0.01, weight_decay=5e-4):
        self.adj = adj.to(device)
        self.features = features.to(device)
        self.labels = labels.to(device)
        self.module = module.GCN(
            in_features=features.shape[1],
            hidden=[16],
            out_features=labels.max().item() + 1,
            dropout=dropout
        ).to(device)
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)

        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
    
    def train(self, epochs=200):
        self.module.train()
        for epoch in range(epochs):
            logits = self.module(self.features, self.adj)
            loss = F.cross_entropy(
                logits[self.idx_train],
                self.labels[self.idx_train]
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evalute(self):
        self.module.eval()
        logits = self.module(self.features, self.adj)
        logits = logits.detach().cpu().numpy()
        y_pred = np.argmax(logits, axis=1)
        return y_pred