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


def loaders(device, batch_size=1):
    settings = configparser.ConfigParser()
    settings_dir = os.path.join(get_root(), 'dataset', 'hcp', 'res', 'hcp_database.ini')
    settings.read(settings_dir)

    params = configparser.ConfigParser()
    params_dir = os.path.join(get_root(), 'dataset', 'hcp', 'res', 'hcp_experiment.ini')
    params.read(params_dir)

    #TODO create logger(settings[logging_level], datefmt=)
    # logging.getlogger()

    train_set = HcpDataset(device, settings, params, test=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)

    test_set = HcpDataset(device, settings, params, test=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


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

    def __init__(self, device, settings, params, coarsening_levels=1, test=False):

        self.params = params
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

        self.H = int(params['TIME_SERIES']['horizon'])
        self.Gp = int(params['TIME_SERIES']['guard_front'])
        self.Gn = int(params['TIME_SERIES']['guard_back'])

        self.transform = SlidingWindow(self.H, self.Gp, self.Gn)
        self.device = device

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):

        # file = os.path.join(self.data_path, self.subjects[idx])

        subject = self.subjects[idx]

        data = process_subject(self.params, subject, [self.session], self.loaders)

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

        Xw, yoh = self.transform(cues, ts, perm)

        return Xw, yoh, coos, perm


class SlidingWindow(object):

    def __init__(self, H, Gp, Gn):
        self.H = H
        self.Gp = Gp
        self.Gn = Gn

    def __call__(self, cues, ts, perm):

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
            num_examples = Np * N

            for i in range(Np):
                for j in range(m):
                    temp_idx = [idx for idx, e in enumerate(C[i, j, :]) if e == 1]
                    cue_idx1 = [idx - Gn for idx in temp_idx]
                    cue_idx2 = [idx + Gp for idx in temp_idx]
                    cue_idx = list(zip(cue_idx1, cue_idx2))

                    for idx in cue_idx:
                        C_temp[slice(*idx)] = j + 1

                y[i, :] = C_temp[0: N]

            y = np.reshape(y, num_examples)
            k = np.max(np.unique(y))
            yoh = one_hot(y, k + 1)

            return yoh

        def encode_x(X):
            X_windowed = []
            X = X.astype('float32')
            M, Q = X[0].shape
            Mnew = len(perm)
            assert Mnew >= M

            if Mnew > M:
                X = pad(X)

            for t in range(N):
                X_windowed.append(X[0, perm, t: t + self.H])  # reorder the nodes based on perm order

            return X_windowed

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

        C = np.expand_dims(cues, 0)
        X = np.expand_dims(ts, 0)

        _, m, _ = C.shape
        Np, p, T = X.shape
        N = T - self.H + 1

        yoh = encode_y(C, Np, N, T, m, self.Gn, self.Gp)

        X_windowed = encode_x(X)

        return X_windowed, yoh
