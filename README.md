Curvature Graph Neural Network
==========================================
A PyTorch implementation of Curvature Graph Neural Network

This is a PyTorch implementation of [Curvature Graph Neural Network](https://www.sciencedirect.com/science/article/pii/S0020025521012986) which has been published by Information Science.

You can visit the document at [sciencedirect](https://www.sciencedirect.com/science/article/pii/S0020025521012986) or [arXiv]( https://arxiv.org/abs/2106.15762).

This repo is also a part of our work on graph representation learning and its application for Temporal-Spatial Prediction, called [T-GCN](https://github.com/lehaifeng/T-GCN), from a geometry perspective.

### Abstract
<p align="justify">
Graph neural networks (GNNs) have achieved great success in many graph-based tasks. Much work is dedicated to empowering GNNs with adaptive locality ability, which enables the measurement of the importance of neighboring nodes to the target node by a node-specific mechanism. However, the current node-specific mechanisms are deficient in distinguishing the importance of nodes in the topology structure. We believe that the structural importance of neighboring nodes is closely related to their importance in aggregation. In this paper, we introduce discretegraph curvature (the Ricci curvature) to quantify the strength of the structural connection of pair-wise nodes. We propose a Curvature Graph Neural Network (CGNN), which effectively improves the adaptive locality ability of GNNs by leveraging the structural properties of graph curvature. To improve the adaptability of curvature on various datasets, we explicitly transform curvature into the weights of neighboring nodes by the necessary Negative Curvature Processing Module and Curvature Normalization Module. Then, we conduct numerous experiments on various synthetic and real-world datasets. The experimental results on synthetic datasets show that CGNN effectively exploits the topology structure information and that the performance is significantly improved. CGNN outperforms the baselines on 5 dense node classification benchmark datasets. This study deepens the understanding of how to utilize advanced topology information and the importance of neighboring nodes from the perspective of graph curvature and encourages us to bridge the gap between graph theory and neural networks.

---------------------------------------------------

### Datasets
All of datasets is loaded and processed by [Pytorch-Geometric](https://github.com/pyg-team/pytorch_geometric). Note that the version of Pytorch-Geometric is `1.5.0`, which has a slight difference with the latest version on loading these dataset.  
The Ricci Curvature of these datasets is saved on `data/Ricci`. To compute curvature, please refer to the Python library [GraphRicciCurvature](https://github.com/saibalmars/GraphRicciCurvature). 

### Options

Training the model is handled by the `main.py` script which provides the following command line arguments.  

```
  --data_path        STRING    Path of saved processed data files.                  Required is False    Default is ./data.
  --dataset          STRING    Name of the datasets.                                Required is True.
  --NCTM             STRING    Type of Negative Curvature Transformation Module.    Required is True     Choices are ['linear', 'exp'].
  --CNM              STRING    Type of Curvature Normalization Module.              Required is True     Choices are ['symmetry-norm', '1-hop-norm', '2-hop-norm'].
  --d_hidden         INT       Dimension of the hidden node features.               Required is False    Default is 64.
  --epochs           INT       The maximum iterations of training.                  Required is False    Default is 200.
  --num_expriment    INT       The number of the repeating expriments.              Required is False    Default is 50.
  --early_stop       INT       Early stop.                                          Required is False    Default is 20.
  --dropout          FLOAT     Dropout.                                             Required is False    Default is 0.5.
  --lr               FLOAT     Learning rate.                                       Required is False    Default is 0.005.
  --weight_decay     FLOAT     Weight decay.                                        Required is False    Default is 0.0005.
```

### Examples
The following commands learn the weights of a curvature graph neural network.
```commandline
python main.py --dataset Cora --NCTM linear --CNM symmetry-norm
```
Another examples is that the following commands learn the weights of the curvature graph neural network with 2-hop normalization on Citeseer.
```commandline
python main.py --dataset Citeseer --NCTM linear --CNM 2-hop-norm
```
  
### Citation information
If our repo is useful to you, please cite our published paper as follow:
```
Bibtex
@article{li2021cgnn,
    title={Curvature Graph Neural Network},
    author={Li, Haifeng and Cao, Jun and Zhu, Jiawei and Liu, Yu and Zhu, Qing and Wu, Guohua},
    journal={Information Sciences},
    DOI = {10.1016/j.ins.2021.12.077},
    year={2021},
    type = {Journal Article}
}
  
Endnote
%0 Journal Article
%A Li, Haifeng
%A Cao, Jun
%A Zhu, Jiawei
%A Liu, Yu
%A Zhu, Qing
%A Wu, Guohua
%D 2021
%T Curvature Graph Neural Network
%B Information Sciences
%R 10.1016/j.ins.2021.12.077
%! Curvature Graph Neural Network

```
