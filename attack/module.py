import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv

class GCN(nn.Module):
    def __init__(self, in_features, hidden, out_features, dropout=0.5):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, hidden[0]))
        for i in range(1, len(hidden)):
            self.layers.append(nn.Linear(hidden[i - 1], hidden[i]))
        
        self.out_layer = nn.Linear(hidden[-1], out_features)
        self.dropout = dropout

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(adj @ x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_layer(adj @ x)
        return x
    
class GAT(nn.Module):
    def __init__(self, in_features, hidden, out_features, heads=8, dropout=0.5):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_features, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, out_features, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv1(x, adj)
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, adj)
        return x

class GraphSAGE(nn.Module):
    def __init__(self, in_features, hidden, out_features, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_features, hidden)
        self.conv2 = SAGEConv(hidden, out_features)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x