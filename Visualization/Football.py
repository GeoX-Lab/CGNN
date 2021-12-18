import urllib.request as urllib
import io
import zipfile
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


seed = 2020
url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"
sock = urllib.urlopen(url)  # open URL
s = io.BytesIO(sock.read())  # read into BytesIO "file"
sock.close()

zf = zipfile.ZipFile(s)  # zipfile object
txt = zf.read("football.txt").decode()  # read info file
gml = zf.read("football.gml").decode()  # read gml data
# throw away bogus first line with # from mejn files
gml = gml.split("\n")[1:]
G = nx.parse_gml(gml)  # parse gml data

nodes_list = list(G.nodes())
nodes_label = np.ones(nx.number_of_nodes(G), dtype=np.int)*(-1)
new_G = nx.Graph()
for n1,n2 in G.edges():
    new_n1 = nodes_list.index(n1)
    new_n2 = nodes_list.index(n2)
    new_G.add_edge(new_n1, new_n2)
    nodes_label[new_n1], nodes_label[new_n2] = G.nodes[n1]['value'], G.nodes[n2]['value']

X = nx.adjacency_matrix(new_G)
X = np.array(X.todense())

train_mask = np.zeros(nx.number_of_nodes(G), dtype=np.int)
com = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: []}
for i in range(nodes_label.shape[0]):
    com[nodes_label[i]].append(i)
for i in range(len(com)):
    np.random.seed(2020)
    ind = np.random.choice(com[i], 1)
    train_mask[ind] = 1
test_mask = 1 - train_mask

for n1, n2 in list(new_G.edges()):
    new_G[n1][n2]['weight'] = 1
nx.write_edgelist(new_G, r'./Football/raw/graph', delimiter=',')
np.savetxt(r'./Football/raw/y', nodes_label, fmt='%d')
np.savetxt(r'./Football/raw/X', X, fmt='%d', delimiter=',')
np.savetxt(r'./Football/raw/train_mask', train_mask, fmt='%d')
np.savetxt(r'./Football/raw/test_mask', test_mask, fmt='%d')
# print(len(list(new_G.neighbors(24))))
# print(len(list(new_G.neighbors(69))))
# print(len(list(new_G.neighbors(11))))
# print(len(list(new_G.neighbors(50))))
# print(len(list(new_G.neighbors(90))))
# print(len(list(new_G.neighbors(28))))