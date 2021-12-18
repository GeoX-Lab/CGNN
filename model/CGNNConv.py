import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Sequential as seq, Parameter, LeakyReLU, init, Linear
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import softmax, degree
import os
import numpy as np


class CGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, weight_agg, p=0.6, bias=True, **kwargs):
        super(CGNNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.dropout = p
        self.out_channels = out_channels
        self.weight_agg = weight_agg
        self.lin = Linear(in_channels, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        weight_agg = F.dropout(self.weight_agg, p=self.dropout, training=self.training)
        return weight_agg.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 5: Return new node embeddings.
        return aggr_out
