import torch
from torch import nn

class GCN(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super(GCN, self).__init__()
        self.conv1 = nn.Linear(in_features, hidden[0])
        for i in range(1, len(hidden)):
            setattr(self, "conv"+str(i+1), nn.Linear(hidden[i-1], hidden[i]))
        self.conv2 = nn.Linear(hidden[-1], out_features)

    def forward(self):
        pass