import numpy as np
import scipy.sparse as sp
from pygcn.utils import normalize

'''测试论文编号处理'''
# 读取原始数据集
path="../data/cora/"
dataset = "cora"
idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))

# build graph
# idx_features_labelsidx_features_labels[:, 0]表示取论文编号
idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

# 通过建立论文序号的序列，得到论文序号的字典
idx_map = {j: i for i, j in enumerate(idx)}

# 读取图的边（论文间的引用关系）
# cora.cites共5429行， 每一行有两个论文编号，表示第一个编号的论文先写，第二个编号的论文引用第一个编号的论文。
edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)

# 进行一次论文序号的映射
# 论文编号没有用，需要重新的其进行编号（从0开始），然后对原编号进行替换。
# 所以目的是把离散的原始的编号，变成0 - 2707的连续编号
edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

# coo_matrix()：系数矩阵的压缩。分别定义有那些非零元素，以及各个非零元素对应的row和col，最后定义稀疏矩阵的shape。
adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(2708, 2708),
                        dtype=np.float32)

# build symmetric adjacency matrix
adj_sysm = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

adj_sysm_self= adj_sysm + sp.eye(adj.shape[0])

adj_norm = normalize(adj_sysm_self)