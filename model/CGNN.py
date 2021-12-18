import torch
import torch.nn.functional as F
from model.CGNNConv import CGNNConv
from torch_geometric.utils import add_self_loops, remove_self_loops
import numpy as np
import random
from torch_scatter import scatter_add
import copy


class CGNN(torch.nn.Module):
    def __init__(self, d_input, d_output, w_mul, d_hidden=64, p=0.0, **kwargs):
        super(CGNN, self).__init__()
        self.conv1 = CGNNConv(d_input, d_hidden, w_mul, p=p, **kwargs)
        self.conv2 = CGNNConv(d_hidden, d_output, w_mul, p=p, **kwargs)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index,
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)


def cal_GCN_weight(edge_index, edge_weight=None, dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def map_ricci(filename, dataset, num_nodes, map_type='linear', edge_index=None):
    f = open(filename)
    cur_list = list(f)
    if dataset in ['WikiCS']:    # directed graph
        ricci_cur = [[] for i in range(len(cur_list))]
        for i in range(len(cur_list)):
            ricci_cur[i] = [num(s) for s in cur_list[i].split(' ', 2)]
    else:
        ricci_cur = [[] for _ in range(2 * len(cur_list))]
        for i in range(len(cur_list)):
            ricci_cur[i] = [num(s) for s in cur_list[i].split(' ', 2)]
            ricci_cur[i + len(cur_list)] = [ricci_cur[i][1], ricci_cur[i][0], ricci_cur[i][2]]
    ricci_cur = sorted(ricci_cur)
    w_mul = [i[2] for i in ricci_cur] + [1 for i in range(num_nodes)]
    w_mul = np.array(w_mul)
    if map_type == 'linear':
        min_ricci_weight = 0.0    # set the value of epsilon
        min_w_mul = abs(min(w_mul)) + min_ricci_weight
        w_mul = w_mul+min_w_mul

    elif map_type == 'exp':
        w_mul = sigmoid(w_mul)
    else:
        pass
    return w_mul


def call(data, arg, d_input, d_output, **kwargs):
    d_hidden, dropout = 64, arg.dropout
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    filename = r'./data/Ricci/graph_' + arg.dataset + '.edge_list'
    ricci = map_ricci(filename, arg.dataset, data.num_nodes, map_type=arg.NCTM, edge_index=data.edge_index)
    ricci = torch.tensor(ricci, dtype=torch.float)
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
    if arg.CNM == 'symmetry-norm':
        w_mul = cal_GCN_weight(data.edge_index, ricci)
    elif arg.CNM == '1-hop-norm':
        w_mul = ricci / (scatter_add(ricci, data.edge_index[1])[data.edge_index[1]] + 1e-16)
    elif arg.CNM == '2-hop-norm':
        w_mul = ricci / (scatter_add(ricci, data.edge_index[0])[data.edge_index[0]] + 1e-16)
    else:
        w_mul = ricci
    w_mul = torch.tensor(w_mul, dtype=torch.float).to(device)
    model = CGNN(d_input, d_output, w_mul, d_hidden=d_hidden, p=dropout, **kwargs)
    data = data.to(device)
    # model = model.to(device)
    model.to(device).reset_parameters()
    return data, model
