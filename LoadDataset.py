import torch
from torch_geometric.data import InMemoryDataset
import networkx as nx
from torch_geometric.utils import from_networkx, is_undirected, to_networkx
from torch_geometric.io import read_txt_array
from torch_geometric.data import Data
import os.path as osp
import numpy as np


# 将数据整理为torch_geometric的数据输入格式
# raw存放原始的数据集，graph代表图的边信息，x代表节点特征，y代表节点类别，*_mask为对应的训练集、验证集、测试集
# processed存放torch_geometric可直接读取的数据集
class TranductiveDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TranductiveDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data = self.get_data()

        if self.pre_filter is not None:
            # data_list = [data for data in data_list if self.pre_filter(data)]
            data = self.pre_filter(data)

        if self.pre_transform is not None:
            # data_list = [self.pre_transform(data) for data in data_list]
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def get_data(self):
        pass

    def read_file(self, path, name):
        pass


if __name__ == '__main__':
    
    dataset = 'Physics'
    data = TranductiveDataset(r'./data/{}'.format(dataset))[0]
    G = to_networkx(data, to_undirected=True, remove_self_loops=True)
    # print(torch.sum(data.train_mask))
    print(nx.info(G))
