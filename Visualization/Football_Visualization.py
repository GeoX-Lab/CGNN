import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from networkx.algorithms import community
import random
import io
import zipfile
import urllib.request as urllib


def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def com_postion(size, scale=1, center=(0, 0), dim=2):
    # generat the postion for each nodes in a community
    num = size
    center = np.asarray(center)
    theta = np.linspace(0, 1, num + 1)[:-1] * 2 * np.pi
    theta = theta.astype(np.float32)
    pos = np.column_stack([np.cos(theta), np.sin(theta), np.zeros((num, 0))])
    pos = scale * pos + center
    return pos


def node_postion(one_com, scale=1, center=(0, 0), dim=2):
    # generat the postion for each nodes in a community
    num = len(one_com)
    node = list(one_com)
    center = np.asarray(center)
    theta = np.linspace(0, 1, num + 1)[:-1] * 2 * np.pi
    theta = theta.astype(np.float32)
    pos = np.column_stack([np.cos(theta), np.sin(theta), np.zeros((num, 0))])
    pos = scale * pos + center
    pos = dict(zip(node, pos))
    return pos


url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"

sock = urllib.urlopen(url)  # open URL
s = io.BytesIO(sock.read())  # read into BytesIO "file"
sock.close()

zf = zipfile.ZipFile(s)  # zipfile object
txt = zf.read("football.txt").decode()  # read info file
gml = zf.read("football.gml").decode()  # read gml data
gml = gml.split("\n")[1:]
G = nx.parse_gml(gml)  # parse gml data

com = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: []}
for node in G.nodes():
    ind = int(G.nodes[node]['value'])
    com[ind].append(node)

# adjust the index of communities 5-1, 10-0
com[5], com[1] = com[1], com[5]
com[10], com[0] = com[0], com[10]
# adjust the index of nodes {11: (BoiseState-FresnoState), 0: (NewMexicoState-LouisianaLafayette)}
com[0][1], com[0][6] = com[0][6], com[0][1]
com[0][0], com[0][1] = com[0][1], com[0][0]
com[11][0], com[11][1] = com[11][1], com[11][0]
com[11][1], com[11][2] = com[11][2], com[11][1]
com[11][3], com[11][5] = com[11][5], com[11][3]

num_com = len(com)
# find intra_com links
intra_links = {}
for i in range(num_com):
    intra_links[i] = []

for link in nx.edges(G):
    for i in range(num_com):
        if (link[0] in com[i]) & (link[1] in com[i]):
            intra_links[i].append(link)

com_center = com_postion(num_com, scale=5)  # print(com_center)
pos = dict()
for val in range(num_com):
    node_pos = node_postion(com[val], scale=0.8, center=com_center[val])
    pos.update(node_pos)

# 根据里奇曲率画边的颜色
node_list = list(G.nodes())
ricci = np.loadtxt(r'./Ricci/graph_Football.edge_list', delimiter=' ')
for row in ricci:
    n1, n2, r = int(row[0]), int(row[1]), row[2]
    n1, n2 = node_list[n1], node_list[n2]
    G[n1][n2]['weight'] = r
weight = [G[n1][n2]['weight'] for n1,n2 in G.edges()]

# 根据类别确定节点的颜色
model_name = 'CurvGN'
colors = ['#f77f00', '#9d4edd', '#40916c', '#ffd6a5', '#606c38', '#a8dadc', '#c7cfb7',
          '#e5383b', '#8e9aaf', '#0353a4', '#ffafcc', '#bdb2ff']
# label = np.loadtxt(r'D:\RicciGN\Visualization\Football\raw\y', dtype=np.int)
label = np.loadtxt(r'D:\RicciGN\Visualization\Predict\{}'.format(model_name), dtype=np.int)
colors_list = [colors[i] for i in label]

options = {'font_family': 'serif', 'font_weight': 'semibold', 'font_size': '6', 'font_color': '#212529'}
plt.figure(figsize=(10, 10))
nx.draw(G, pos, with_labels=True, node_size=15, edgelist=[], **options)

# 画节点
nx.draw_networkx_nodes(G, pos, node_size=150, alpha=1, node_color=colors_list)
# for val in range(12):
#     if val == 5 or val == 9:
#         continue
#     nx.draw_networkx_nodes(G, pos, node_size=150, alpha=1, nodelist=list(com[val]), node_color=colors[val])
# for val in [5, 9]:
#     nx.draw_networkx_nodes(G, pos, node_size=150, alpha=0.4, nodelist=list(com[val]), node_color=colors[val])

# 画边
nx.draw_networkx_edges(G, pos, alpha=1, width=0.8,
                       edge_color=weight, edge_cmap=plt.get_cmap('coolwarm'), edge_vmin=-1, edge_vmax=1)  # seismic

# nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.8, edge_color='grey')
# for val in range(num_com):
#     nx.draw_networkx_edges(G, pos, alpha=0.5, edgelist=intra_links[val], width=0.8, edge_color='black')

plt.axis("off")
# plt.savefig('./Image/Origin_edge_grey.png', dpi=600)
plt.savefig('./Image/{}_edge_grey.png'.format(model_name), dpi=600)
plt.show()
