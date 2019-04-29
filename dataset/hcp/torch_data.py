import os
import torch
import configparser

import numpy as np
import scipy.io as sio

from util.path import get_root
import ext.gcn.coarsening as coarsening

from util.encode import one_hot

from dataset.hcp.data import load_subjects, process_subject
from dataset.hcp.matlab_data import get_cues, get_bold, load_structural, encode
from dataset.hcp.downloaders import DtiDownloader, HcpDownloader


def loaders(device, parcellation, batch_size=1):
    settings = configparser.ConfigParser()
    settings_dir = os.path.join(get_root(), 'dataset', 'hcp', 'res', 'hcp_database.ini')
    settings.read(settings_dir)

    #TODO create logger(settings[logging_level], datefmt=)
    # logging.getlogger()

    train_set = HcpDataset(device, settings, parcellation, test=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)

    test_set = HcpDataset(device, settings, parcellation, test=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class MatlabDataset(torch.utils.data.Dataset):  #TODO Delete and commit

    def __init__(self, perm):
        self.list_file = 'subjects_test.txt'
        list_url = os.path.join(get_root(), 'conf', self.list_file)
        self.data_path = os.path.join(os.path.expanduser("~"), 'data_full')

        self.subjects = load_subjects(list_url)
        post_fix = '_aparc_tasks_aparc.mat'
        self.filenames = [s + post_fix for s in self.subjects]

        self.p = 148
        self.T = 284
        self.session = 'MOTOR_LR'

        self.transform = Encode(15, 4, 4, perm)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file = os.path.join(self.data_path, self.filenames[idx])
        ds = sio.loadmat(file).get('ds')
        MOTOR = ds[0, 0][self.session]

        C_i = np.expand_dims(get_cues(MOTOR), 0)
        X_i = np.expand_dims(get_bold(MOTOR).transpose(), 0)

        Xw, yoh = self.transform(C_i, X_i)

        return Xw.astype('float32'), yoh


class StreamMatlabDataset(torch.utils.data.Dataset):

    def __init__(self):
        normalized_laplacian = True
        coarsening_levels = 4

        list_file = 'subjects_inter.txt'
        list_url = os.path.join(get_root(), 'conf', list_file)
        subjects_strut = load_subjects(list_url)

        structural_file = 'struct_dti.mat'
        structural_url = os.path.join(get_root(), 'load', 'hcpdata', structural_file)
        S = load_structural(subjects_strut, structural_url)
        S = S[0]

        # avg_degree = 7
        # S = scipy.sparse.random(65000, 65000, density=avg_degree/65000, format="csr")

        self.graphs, self.perm = coarsening.coarsen(S, levels=coarsening_levels, self_connections=False)

        self.list_file = 'subjects_hcp_all.txt'
        list_url = os.path.join(get_root(), 'conf', self.list_file)
        self.data_path = os.path.join(os.path.expanduser("~"), 'data_full')

        self.subjects = load_subjects(list_url)
        post_fix = '_aparc_tasks_aparc.mat'
        self.filenames = [s + post_fix for s in self.subjects]

        self.p = 148  # 65000
        self.T = 284
        self.session = 'MOTOR_LR'

        self.transform = SlidingWindow(15, 4, 4)

    def get_graphs(self, device):
        coos = [torch.tensor([graph.tocoo().row, graph.tocoo().col], dtype=torch.long).to(device) for graph in
                self.graphs]
        return self.graphs, coos, self.perm

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file = os.path.join(self.data_path, self.filenames[idx])
        ds = sio.loadmat(file).get('ds')
        MOTOR = ds[0, 0][self.session]

        C_i = np.expand_dims(get_cues(MOTOR), 0)
        X_i = np.expand_dims(get_bold(MOTOR).transpose(), 0)

        # X_i = np.random.rand(1, 65000, 284)

        Xw, yoh = self.transform(C_i, X_i, self.perm)

        return Xw, yoh


class HcpDataset(torch.utils.data.Dataset):

    def __init__(self, device, settings, parcellation='aparc', coarsening_levels=1, test=False):

        self.parcellation = parcellation
        self.settings = settings

        hcp_downloader = HcpDownloader(settings)
        dti_downloader = DtiDownloader(settings)
        self.loaders = [hcp_downloader, dti_downloader]

        self.list_file = 'subjects.txt'
        if test:
            list_url = os.path.join(get_root(), 'conf/hcp/test/motor_lr', self.list_file)
        else:
            list_url = os.path.join(get_root(), 'conf/hcp/train/motor_lr', self.list_file)

        # self.data_path = os.path.join(expanduser("~"), 'data_dense')

        self.subjects = load_subjects(list_url)

        # TODO session shouldn't be hardcoded
        self.session = 'MOTOR_LR'
        self.coarsening_levels = 1

        # TODO magic numbers
        self.p = 148
        self.T = 284
        self.H = 15
        self.Gp = 4
        self.Gn = 4

        self.transform = SlidingWindow(self.H, self.Gp, self.Gn)
        self.device = device

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):

        # file = os.path.join(self.data_path, self.subjects[idx])

        subject = self.subjects[idx]

        data = process_subject(self.parcellation, subject, [self.session], self.loaders)

        cues = data['functional']['MOTOR_LR']['cues']
        ts = data['functional']['MOTOR_LR']['ts']
        S = data['adj']

        if self.coarsening_levels == 1: #TODO wrap the graph thing in a function, perhaps process_subject
            graphs = [S]
            perm = list(range(0, S.shape[0]))
        else:
            graphs, perm = coarsening.coarsen(S, levels=self.coarsening_levels, self_connections=False)

        coos = [torch.tensor([graph.tocoo().row, graph.tocoo().col], dtype=torch.long).to(self.device) for graph in
                graphs]

        C_i = np.expand_dims(cues, 0)   #TODO put this into transform
        X_i = np.expand_dims(ts, 0)

        Xw, yoh = self.transform(C_i, X_i, perm)

        return Xw, yoh, coos, perm


class Encode(object):   #TODO Delete and commit

    def __init__(self, H, Gp, Gn, perm):
        self.H = H
        self.Gp = Gp
        self.Gn = Gn

    def __call__(self, C, X, perm):
        Xw, y = encode(C, X, self.H, self.Gp, self.Gn)
        Xw = perm_data_time(Xw, perm)

        k = np.max(np.unique(y))
        yoh = one_hot(y, k + 1)

        return Xw, yoh


class SlidingWindow(object):

    def __init__(self, H, Gp, Gn):
        self.H = H
        self.Gp = Gp
        self.Gn = Gn

    def __call__(self, C, X, perm): #TODO: put encode perm function here as two functions, encode_y and encode_x
        Xw, y = encode_perm(C, X, self.H, self.Gp, self.Gn, perm)

        k = np.max(np.unique(y))    #TODO: move within encode_y
        yoh = one_hot(y, k + 1)

        return Xw, yoh


###################### THESE FUNCTIONS NEED TO BE EXPLAINED #####################

def encode_y(C, Np, N, T, m, Gn, Gp):
    """
    Encodes the target signal to account for windowing
    :param C: targets
    :param Np: number of examples
    :param N: length of windowed time signal
    :param T: length of original time signal
    :param m: number of classes
    :param Gn: front guard length
    :param Gp: back guard length
    :return: y: encoded target signal
    """
    y = np.zeros([Np, N])
    C_temp = np.zeros(T)

    for i in range(Np):
        for j in range(m):
            temp_idx = [idx for idx, e in enumerate(C[i, j, :]) if e == 1]
            cue_idx1 = [idx - Gn for idx in temp_idx]
            cue_idx2 = [idx + Gp for idx in temp_idx]
            cue_idx = list(zip(cue_idx1, cue_idx2))

            for idx in cue_idx:
                C_temp[slice(*idx)] = j + 1

        y[i, :] = C_temp[0: N]

    return y


def pad(X, Mnew, M):
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


# TODO can't these functions reuse from (a generalized) data.encode?
# TODO this code is terribly complicated

def encode_perm(C, X, H, Gp, Gn, indices):
    """
    Encodes the time signal and targets to apply the windowing algorithm
    :param C: data labels
    :param X: data to be windowed
    :param H: window size
    :param Gp: start point guard
    :param Gn: end point guard
    :param indices: ordering of graph vertices
    :return: X_windowed, y: encoded time signal and target classes
    """
    _, m, _ = C.shape
    Np, p, T = X.shape
    N = T - H + 1
    num_examples = Np * N

    y = encode_y(C, Np, N, T, m, Gn, Gp)
    y = np.reshape(y, num_examples)

    X_windowed = []
    X = X.astype('float32')
    M, Q = X[0].shape
    Mnew = len(indices)
    assert Mnew >= M

    if Mnew > M:
        X = pad(X)

    for t in range(N):
        X_windowed.append(X[0, indices, t: t + H])  # reorder the nodes based on perm order

    return [X_windowed, y]


def perm_data_time(x, indices): #TODO Delete and commit
    """
    Permute data matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return x

    N, M, Q = x.shape
    Mnew = len(indices)

    assert Mnew >= M

    xnew = np.empty((N, Mnew, Q), dtype="float32")

    for i, j in enumerate(indices):
        # Existing vertex, i.e. real data.
        if j < M:
            xnew[:, i, :] = x[:, j, :]
        # Fake vertex because of singeltons.
        # They will stay 0 so that max pooling chooses the singelton.
        # Or -infty ?
        else:
            xnew[:, i, :] = np.zeros((N, Q))

    return xnew
