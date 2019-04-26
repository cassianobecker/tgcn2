import scipy.sparse as sp
import random
import numpy as np

import ext.gcn.graph as graph


def grid_graph(m, metric='euclidean', number_edges=8, corners=False, shuffled=True):
    z = graph.grid(m)
    dist, idx = graph.distance_sklearn_metrics(z, k=number_edges, metric=metric)
    adj = graph.adjacency(dist, idx)

    if shuffled:
        bdj = adj.toarray()
        bdj = list(bdj[np.triu_indices(adj.shape[0])])
        random.shuffle(bdj)
        adj = np.zeros((adj.shape[0], adj.shape[0]))
        indices = np.triu_indices(adj.shape[0])
        adj[indices] = bdj
        adj = adj + adj.T - np.diag(adj.diagonal())
        adj = sp.csr_matrix(adj)

    # Connections are only vertical or horizontal on the grid.
    # Corner vertices are connected to 2 neighbors only.
    if corners:
        adj = adj.toarray()
        adj[adj < adj.max() / 1.5] = 0
        adj = sp.csr_matrix(adj)
        print('{} edges'.format(adj.nnz))

    # print("{} > {} edges".format(adj.nnz // 2, number_edges * m ** 2 // 2))
    return adj
