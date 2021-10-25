# -*- coding:utf-8 -*-

import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
from utils import load_data
from model import CGNN
import argparse


def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    pred = model(data)
    loss = F.nll_loss(pred[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


def val(data, model):
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    val_mask = data.val_mask
    accs.append(F.nll_loss(model(data)[val_mask], data.y[val_mask]))
    return accs


def main(args, d_input, d_output):
    test_acc_list = []
    for i in range(args.num_expriment):
        data = load_data(args.data_path, args.dataset)
        data, model = globals()[args.model].call(data, args, d_input, d_output)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val_acc = test_acc = 0.0
        best_val_loss = np.inf
        wait_step = 0
        ##########################
        val_loss_list = []
        tem_test_acc_list = []
        for epoch in range(0, args.epochs):
            train(data, model, optimizer)
            train_acc, val_acc, tmp_test_acc, val_loss = val(data, model)
            ##########################
            val_loss_list.append(val_loss.item())
            tem_test_acc_list.append(tmp_test_acc)
            if val_acc >= best_val_acc or val_loss <= best_val_loss:
                if val_acc >= best_val_acc:
                    test_acc = tmp_test_acc
                    early_val_acc = val_acc
                    early_val_loss = val_loss
                best_val_acc = np.max((val_acc, best_val_acc))
                best_val_loss = np.min((val_loss, best_val_loss))
                wait_step = 0
            else:
                wait_step += 1
                if wait_step == args.early_stop:
                    print('Early stop! Min loss: ', best_val_loss, ', Max accuracy: ', best_val_acc)
                    print('Early stop model validation loss: ', early_val_loss, ', accuracy: ', early_val_acc)
                    break
        log = 'Model_type: {}, Dateset_name: {}, Experiment: {:03d}, Test: {:.6f}'
        print(log.format(args.model_type, args.dataset, i + 1, test_acc))
        test_acc_list.append(test_acc * 100)
    log = 'Model_type: {}, Dateset_name: {}, Experiments: {:03d}, Mean: {:.6f}, std: {:.6f}\n'
    print(log.format(args.model_type, args.dataset, args.num_expriment, np.mean(test_acc_list),
                     np.std(test_acc_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CGNN')
    parser.add_argument('--data_path', type=str, help="Path of saved processed data files.", default='./data')
    parser.add_argument('--dataset', type=str, help="Name of the datasets", required=True)
    parser.add_argument('--NCTM', type=str, choices=['linear', 'exp'],
                        help="Type of Negative Curvature Transformation Module", required=True)
    parser.add_argument('--CNM', type=str, choices=['symmetry-norm', '1-hop-norm', '2-hop-norm'],
                        help="Type of Curvature Normalization Module", required=True)
    parser.add_argument('--d_hidden', type=int, help="Dimension of the hidden node features", default=64)
    parser.add_argument('--epochs', type=int, help="The maximum iterations of training", default=200)
    parser.add_argument('--num_expriment', type=int, help="The number of the repeating expriments", default=50)
    parser.add_argument('--early_stop', type=int, help="Early stop", default=20)
    parser.add_argument('--dropout', type=float, help="Dropout", default=0.5)
    parser.add_argument('--lr', type=float, help="Learning rate", default=0.005)
    parser.add_argument('--weight_decay', type=float, help="Weight decay", default=0.0005)
    args = parser.parse_args()

    datasets_config = {
        'Cora': {'d_input': 1433,
                 'd_output': 7},
        'Citeseer': {'d_input': 3703,
                     'd_output': 6},
        'PubMed': {'d_input': 500,
                   'd_output': 3},
        'CS': {'d_input': 6805,
               'd_output': 15},
        'Physics': {'d_input': 8415,
                    'd_output': 5},
        'Computers': {'d_input': 767,
                      'd_output': 10},
        'Photo': {'d_input': 745,
                  'd_output': 8},
        'WikiCS': {'d_input': 300,
                   'd_output': 10},
    }

    args.model = 'CGNN'
    args.model_type = 'CGNN_{}_{}_{}'.format(args.NCTM, args.CNM, args.dropout)
    d_input, d_output = datasets_config[args.dataset]['d_input'], datasets_config[args.dataset]['d_output']
    main(args, d_input, d_output)
