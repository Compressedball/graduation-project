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
    
class GATSprase():
    def __init__(self, edge_index, features, labels,
                 idx_train, idx_val, idx_test,
                 device,
                 dropout=0.5, lr=0.01, weight_decay=5e-4):
        self.edge_index = edge_index.to(device)
        self.features = features.to(device)
        self.labels = labels.to(device)
        self.module = module.GAT(
            in_features=features.shape[1],
            hidden=16,
            out_features=labels.max().item() + 1,
            heads=8,
            dropout=dropout
        ).to(device)
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)

        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

    def train(self, epochs=200):
        self.module.train()
        for epoch in range(epochs):
            logits = self.module(self.features, self.edge_index)
            loss = F.cross_entropy(
                logits[self.idx_train],
                self.labels[self.idx_train]
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evalute(self):
        self.module.eval()
        logits = self.module(self.features, self.edge_index)
        logits = logits.detach().cpu().numpy()
        y_pred = np.argmax(logits, axis=1)
        return y_pred
    

class GraphSAGESprase():
    def __init__(self, edge_index, features, labels,
                 idx_train, idx_val, idx_test,
                 device,
                 dropout=0.5, lr=0.01, weight_decay=5e-4):
        self.edge_index = edge_index.to(device)
        self.features = features.to(device)
        self.labels = labels.to(device)
        self.module = module.GraphSAGE(
            in_features=features.shape[1],
            hidden=16,
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
            logits = self.module(self.features, self.edge_index)
            loss = F.cross_entropy(
                logits[self.idx_train],
                self.labels[self.idx_train]
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evalute(self):
        self.module.eval()
        logits = self.module(self.features, self.edge_index)
        logits = logits.detach().cpu().numpy()
        y_pred = np.argmax(logits, axis=1)
        return y_pred
    
def victim_model(adj, features, labels,
                   idx_train, idx_val, idx_test,
                   device,
                   dropout=0.5, lr=0.01, weight_decay=5e-4,
                   model='GCN'):
    
    adj_norm = utils.normalize_sparse_coo_adj(adj)
    features_norm = utils.normalize_features(features)
    edge_index = np.vstack((adj.row, adj.col))

    edge_index_tensor = torch.LongTensor(edge_index)
    adj_norm_tensor = utils.sparse_mx_to_torch_sparse_tensor(adj_norm)
    features_norm_tensor = torch.FloatTensor(features_norm)
    labels_tensor = torch.LongTensor(labels)
    idx_train_tensor = torch.LongTensor(idx_train)
    idx_val_tensor = torch.LongTensor(idx_val)
    idx_test_tensor = torch.LongTensor(idx_test)
    print(type(adj_norm_tensor))

    if model == 'GCN':
        return GCNSprase(adj_norm_tensor, features_norm_tensor, labels_tensor,
                         idx_train_tensor, idx_val_tensor, idx_test_tensor,
                         device,
                         dropout, lr, weight_decay)
    elif model == 'GAT':
        return GATSprase(edge_index_tensor, features_norm_tensor, labels_tensor,
                         idx_train_tensor, idx_val_tensor, idx_test_tensor,
                         device,
                         dropout, lr, weight_decay)
    elif model == 'GraphSAGE':
        return GraphSAGESprase(edge_index_tensor, features_norm_tensor, labels_tensor,
                               idx_train_tensor, idx_val_tensor, idx_test_tensor,
                               device,
                               dropout, lr, weight_decay)
    else:
        raise ValueError(f"Unsupported model type: {model}. Please use model name GCN, GAT, GraphSAGE")