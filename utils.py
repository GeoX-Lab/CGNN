import math
import numpy as np
import torch_geometric.datasets
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch
import os
from LoadDataset import TranductiveDataset


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a ** 2) * fan))
        tensor.data.uniform_(-bound, bound)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def normal(tensor, mean, std):
    if tensor is not None:
        tensor.data.normal_(mean, std)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


# seed默认值与pitfalls of GCN中的相同
def random_split_dataset(node_num, seed=2018,
                         train_size=None, val_size=None, test_size=None):
    random_state = np.random.RandomState(seed)
    remaining_indices = list(range(node_num))

    train_indices = random_state.choice(remaining_indices, train_size, replace=False)
    remaining_indices = np.setdiff1d(remaining_indices, train_indices)
    val_indices = random_state.choice(remaining_indices, val_size, replace=False)
    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)
    return split_to_binary(node_num, train_indices), split_to_binary(node_num, val_indices), split_to_binary(node_num,
                                                                                                             test_indices)


def split_to_binary(node_num, indices):
    data = np.zeros(node_num, dtype=np.bool)
    data[indices] = True
    return data


def sample_per_class(random_state, labels, num_examples_per_class):
    num_classes = max(labels) + 1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for i, label in enumerate(labels):
            if class_index == label:
                sample_indices_per_class[class_index].append(i)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def load_data(data_path, d_name):
    if d_name == 'Cora' or d_name == 'Citeseer' or d_name == 'PubMed':
        d_loader = 'Planetoid'
    elif d_name == 'Computers' or d_name == 'Photo':
        d_loader = 'Amazon'
    elif d_name == 'CS' or d_name == 'Physics':
        d_loader = 'Coauthor'
    else:
        d_loader = 'TranductiveDataset'
    if d_loader == 'Planetoid':
        data = getattr(torch_geometric.datasets, d_loader)(data_path, d_name, transform=T.NormalizeFeatures())[0]
    elif d_loader == 'TranductiveDataset':
        data = TranductiveDataset(os.path.join(data_path, d_name))[0]
    else:
        data = getattr(torch_geometric.datasets, d_loader)(os.path.join(data_path, d_name), d_name)[0]
        # CurvGN中的划分方法
        # index = [i for i in range(len(data.y))]
        # train_len = 20 * int(data.y.max() + 1)
        # train_mask = torch.tensor([i < train_len for i in index])
        # val_mask = torch.tensor([i >= train_len and i < 500 + train_len for i in index])
        # test_mask = torch.tensor([i >= len(data.y) - 1000 for i in index])
        # data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
        labels = data.y.numpy()
        num_nodes = data.num_nodes
        seed = 2018
        random_state = np.random.RandomState(seed)
        remaining_indices = list(range(num_nodes))
        train_indices = sample_per_class(random_state, labels, 20)
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, 500, replace=False)
        remaining_indices = np.setdiff1d(remaining_indices, val_indices)
        test_indices = random_state.choice(remaining_indices, 1000, replace=False)
        train_mask, val_mask, test_mask = split_to_binary(num_nodes, train_indices), \
                                          split_to_binary(num_nodes, val_indices), \
                                          split_to_binary(num_nodes, test_indices)
        data.train_mask, data.val_mask, data.test_mask = torch.tensor(train_mask), \
                                                         torch.tensor(val_mask), \
                                                         torch.tensor(test_mask)
    return data
