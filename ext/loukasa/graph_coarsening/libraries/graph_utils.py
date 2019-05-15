import numpy as np
import pygsp as gsp


def plot_edge_signal(G, signal, edge_width=2, alpha=0.99, node_size=20, size=3, title=' ', ax=None, normalize=True):
    import matplotlib
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.cm as cm
    cmap = cm.RdBu_r

    from matplotlib.colors import Normalize
    if normalize:
        norm = Normalize(vmin=np.min(signal), vmax=np.max(signal))
    else:
        norm = lambda x: x

    if ax is None:
        fig = plt.figure(figsize=(size * 3, size * 2))
        if G.coords.shape[1] == 2:
            ax = fig.add_subplot(1, 1, 1)
        elif G.coords.shape[1] == 3:
            ax = fig.add_subplot(1, 1, 1, projection='3d')

    edges = np.array(G.get_edge_list()[0:2])

    if G.coords.shape[1] == 2:
        ax.axis('off')
        ax.set_title(title)

        [x, y] = G.coords.T
        for eIdx in range(0, edges.shape[1]):
            color = cmap(norm(signal[eIdx]))
            ax.plot(x[edges[:, eIdx]], y[edges[:, eIdx]], color=color, alpha=alpha, linewidth=edge_width)
        ax.scatter(x, y, c='k', s=node_size, alpha=alpha)

    elif G.coords.shape[1] == 3:
        ax.axis('off')
        ax.set_title(title)

        [x, y, z] = G.coords.T
        for eIdx in range(0, edges.shape[1]):
            color = cmap(norm(signal[eIdx]))
            ax.plot(x[edges[:, eIdx]], y[edges[:, eIdx]], zs=z[edges[:, eIdx]], color=color, alpha=alpha,
                    lineWidth=edge_width)
        ax.scatter(x, y, z, c='k', s=node_size, alpha=alpha)


def get_neighbors(G, i):
    return G.A[i, :].indices
    # return np.arange(G.N)[np.array((G.W[i,:] > 0).todense())[0]]


def get_giant_component(G):
    from scipy.sparse import csgraph

    [ncomp, labels] = csgraph.connected_components(G.W, directed=False, return_labels=True)

    W_g = np.array((0, 0))
    coords_g = np.array((0, 2))
    keep = np.array(0)

    for i in range(0, ncomp):

        idx = np.where(labels != i)
        idx = idx[0]

        if G.N - len(idx) > W_g.shape[0]:
            W_g = G.W.toarray()
            W_g = np.delete(W_g, idx, axis=0)
            W_g = np.delete(W_g, idx, axis=1)
            if hasattr(G, 'coords'):
                coords_g = np.delete(G.coords, idx, axis=0)
            keep = np.delete(np.arange(G.N), idx)

    if not hasattr(G, 'coords'):
        # print(W_g.shape)
        G_g = gsp.graphs.Graph(W=W_g)
    else:
        G_g = gsp.graphs.Graph(W=W_g, coords=coords_g)

    return (G_g, keep)


def get_S(G):
    """
    Construct the N x |E| gradient matrix S
    """
    # the edge set
    edges = G.get_edge_list()
    weights = np.array(edges[2])
    edges = np.array(edges[0:2])
    M = edges.shape[1]

    # Construct the N x |E| gradient matrix S
    S = np.zeros((G.N, M))
    for e in np.arange(M):
        S[edges[0, e], e] = np.sqrt(weights[e])
        S[edges[1, e], e] = -np.sqrt(weights[e])

    return S


# Compare the spectum of L and Lc
def eig(A, order='ascend'):
    # eigenvalue decomposition
    [l, X] = np.linalg.eigh(A)

    # reordering indices     
    idx = l.argsort()
    if order == 'descend':
        idx = idx[::-1]

    # reordering     
    l = np.real(l[idx])
    X = X[:, idx]
    return (X, np.real(l))


def zero_diag(A):
    import scipy as sp

    if sp.sparse.issparse(A):
        return A - sp.sparse.dia_matrix((A.diagonal()[sp.newaxis, :], [0]), shape=(A.shape[0], A.shape[1]))
    else:
        D = A.diagonal()
        return A - np.diag(D)


def is_symmetric(As):
    """Check if a sparse matrix is symmetric

    Parameters
    ----------
    As : array or sparse matrix
        A square matrix.

    Returns
    -------
    check : bool
        The check result.

    """
    from scipy import sparse

    if As.shape[0] != As.shape[1]:
        return False

    if not isinstance(As, sparse.coo_matrix):
        As = sparse.coo_matrix(As)

    r, c, v = As.row, As.col, As.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
        return False

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]

    check = np.allclose(vl, vu)

    return check
