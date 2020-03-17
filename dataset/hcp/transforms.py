import random
import numpy as np
from util.encode import one_hot
from ext.loukasa.graph_coarsening.libraries.coarsening_utils import graphs, coarsen
from scipy.sparse import csr_matrix


class TrivialCoarsening(object):

    def __call__(self, weight_matrix, level=2):
        order = list(range(0, weight_matrix.shape[0]))
        eye = np.identity(weight_matrix.shape[0])
        eye = eye[:, order]
        eye_mtx = np.asmatrix(eye)

        graph_list = [self._to_coos(weight_matrix)]
        mapping_list = [[]]
        edge_weight_list = [weight_matrix.data]

        for i in range(level):
            graph_list.append(self._to_coos(weight_matrix))
            mapping_list.append(eye_mtx)
            edge_weight_list.append(weight_matrix.data)

        return graph_list, edge_weight_list, mapping_list

    def _to_coos(self, A):
        return [A.tocoo().row, A.tocoo().col]


class PseudoSpectralCoarsening(object):

    def __call__(self, weight_matrix):    #TODO: SET TO FALSE
        order = list(range(0, weight_matrix.shape[0]))
        eye = np.identity(weight_matrix.shape[0])

        graph_list = [self._to_coos(weight_matrix)]
        mapping_list = [[]]
        edge_weight_list = [weight_matrix.data]

        #random.shuffle(order)
        eye = eye[:, order]
        eye_mtx = np.asmatrix(eye)

        w_temp = csr_matrix(eye) @ weight_matrix
        weight_matrix_2 = w_temp @ csr_matrix(eye).transpose()

        graph_list.append(self._to_coos(weight_matrix_2))
        mapping_list.append(eye_mtx)
        edge_weight_list.append(weight_matrix_2.data)

        return graph_list, edge_weight_list, mapping_list

    def _to_coos(self, A):
        return [A.tocoo().row, A.tocoo().col]


class SpectralCoarsening(object):

    def __init__(self, resolutions):
        self.mtx_default = None
        self.graph_default = None
        self.weights_default = None
        self.mapping_default = None

        self.resolutions = resolutions

    def __call__(self, weight_matrix):
        if self.mtx_default is not None and (self.mtx_default - weight_matrix).nnz == 0:
            return self.graph_default, self.weights_default, self.mapping_default
        else:

            self.mtx_default = weight_matrix
            current_weight_matrix = weight_matrix

            graph_list = [self._to_coos(current_weight_matrix)]
            edge_weight_list = [current_weight_matrix.data]
            mapping_list = [[]]

            for resolution in self.resolutions:
                graph_at_resolution = graphs.Graph(W=current_weight_matrix)
                graph_ratio = 1 - resolution / graph_at_resolution.A.shape[0]

                coarsening_mapping, coarsened_graph, _, _ = coarsen(graph_at_resolution,
                                                                K=10, max_levels=60, r=graph_ratio, method='variation_edges')

                current_weight_matrix = csr_matrix(coarsened_graph.W)

                graph_list.append(self._to_coos(current_weight_matrix))
                edge_weight_list.append(current_weight_matrix.data)
                mapping_list.append(coarsening_mapping.todense())

                self.graph_default = graph_list
                self.weights_default = edge_weight_list
                self.mapping_default = mapping_list

            return graph_list, edge_weight_list, mapping_list

    def coarsen_signal(self, signal, mapping):
        xn = np.dot(signal, mapping)
        # torch.matmul(mapping_list[1].type(dtype=torch.cuda.FloatTensor), x)
        return xn

    def _to_coos(self, A):
        return [A.tocoo().row, A.tocoo().col]


class SlidingWindow(object):
    """
    Applies a sliding window to the BOLD time signal and designates a motor task to each window
    :param horizon: length of sliding window/horizon
    :param guard_front: front guard size
    :param guard_back: back guard size
    """

    def __init__(self, params, coarsen):
        self.horizon = int(params['horizon'])
        self.guard_front = int(params['guard_front'])
        self.guard_back = int(params['guard_back'])
        self.coarsen = coarsen

    def __call__(self, cues, ts, perm):
        X = np.expand_dims(ts, 0)
        X_windowed = self.encode_X(X, perm)

        C = np.expand_dims(cues, 0)
        Y_one_hot = self.encode_Y(C, X.shape)

        return X_windowed, Y_one_hot

    def encode_Y(self, C, X_shape):
        """
        Encodes the target signal to account for windowing
        :param C: targets
        :param X_shape: shape of BOLD data (# examples, # parcels, # time samples)
        :return: Y: encoded target signal
        """
        Np, p, T = X_shape
        N = T - self.horizon + 1

        y = np.zeros([Np, N])
        C_temp = np.zeros(T)
        num_examples = Np * N
        m = C.shape[1]

        for i in range(Np):
            for j in range(m):
                # find indices in the original signal with the task
                temp_idx = [idx for idx, e in enumerate(C[i, j, :]) if e == 1]
                # starting indices of the task in the new signal (calculated with guards)
                cue_idx1 = [idx - self.guard_back for idx in temp_idx]
                # ending indices of the task in the new signal (calculated with guards)
                cue_idx2 = [idx + self.guard_front for idx in temp_idx]
                # pair the tuples to form intervals to assign to specific motor task
                cue_idx = list(zip(cue_idx1, cue_idx2))

                for idx in cue_idx:
                    # assign task to specified interval
                    C_temp[slice(*idx)] = j + 1

            y[i, :] = C_temp[0: N]

        y = np.reshape(y, num_examples)
        k = np.max(np.unique(y))
        yoh = one_hot(y, k + 1)

        return yoh

    def encode_X(self, X, perm):
        """
        Provides a list of memory-efficient windowed views into the BOLD time signal.
        :param X: Signal to be encoded
        :return: X_windowed, the list of windowed views
        """

        # we store the views into the signal in a list
        X_windowed = []
        X = X.astype('float32')
        p, T = X[0].shape
        # resulting time signal will have (time_dimension - window_length + 1) time steps
        N = T - self.horizon + 1

        # p_new = len(perm)
        # assert p_new >= p
        #
        # if p_new > p:
        #     X = self.pad(X)
        #
        # for t in range(N):
        #     # reorder the nodes based on perm order
        #     X_windowed.append(X[0, perm, t: t + self.horizon])

        for t in range(N):
            X_windowed.append(X[0, :, t: t + self.horizon])

        return X_windowed

    def pad(self, X, Mnew, M):
        """
        Pads the data with zeros to account for dummy nodes
        :param X: fMRI data
        :param Mnew: number of nodes in graph (w/ dummies)
        :param M: number of nodes in the original graph (w/o dummies)
        :return: padded data
        """
        diff = Mnew - M
        z = np.zeros((X.shape[0], diff, X.shape[2]), dtype="float32")
        X = np.concatenate((X, z), axis=1)
        return X
