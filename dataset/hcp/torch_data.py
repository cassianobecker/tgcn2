import os
import torch
import configparser

import numpy as np

from util.path import get_root
from util.encode import one_hot
from util.logging import init_loggers

from dataset.hcp.hcp_data import HcpReader


def get_settings():
    """
    Creates a ConfigParser object with server/directory/credentials/logging info from preconfigured directory.
    :return: settings, a ConfigParser object
    """
    settings = configparser.ConfigParser()
    settings_dir = os.path.join(get_root(), 'dataset', 'hcp', 'conf', 'hcp_database.ini')
    settings.read(settings_dir)
    return settings


def get_params():
    """
    Creates a ConfigParser object with parameter info for reading fMRI data.
    :return: params, a ConfigParser object
    """
    params = configparser.ConfigParser()
    params_dir = os.path.join(get_root(), 'dataset', 'hcp', 'conf', 'hcp_experiment.ini')
    params.read(params_dir)
    return params


def loaders(device, batch_size=1):
    """
    Creates train and test datasets and places them in a DataLoader to iterate over.
    """

    settings = get_settings()
    params = get_params()

    init_loggers(settings)

    session = 'MOTOR_LR'

    train_set = HcpDataset(device, settings, params, 'train', session)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)

    test_set = HcpDataset(device, settings, params, 'test', session)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class HcpDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset to host and process the BOLD signal and the associated motor tasks
    """

    def __init__(self, device, settings, params, regime, session):
        self.device = device
        self.params = params
        self.settings = settings
        self.session = session

        self.reader = HcpReader(settings, params)
        self.transform = SlidingWindow(params['TIME_SERIES'])

        list_url = os.path.join(get_root(), 'conf', 'hcp', regime, session, 'subjects.txt')
        self.subjects = self.reader.load_subject_list(list_url)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):

        subject = self.subjects[idx]
        data = self.reader.process_subject(subject, [self.session])

        cues = data['functional'][self.session]['cues']
        ts = data['functional'][self.session]['ts']

        S = data['adjacency']
        graphs = [S]
        perm = list(range(0, S.shape[0]))
        coos = [torch.tensor([graph.tocoo().row, graph.tocoo().col], dtype=torch.long).to(self.device)
                for graph in graphs]

        X_windowed, Y_one_hot = self.transform(cues, ts, perm)

        return X_windowed, Y_one_hot, coos, perm

    def data_shape(self):
        shape = self.reader.get_adjacency(self.subjects[0]).shape[0]
        return shape

    def self_check(self):
        for subject in self.subjects:
            self.reader.process_subject(subject, [self.session])


class SlidingWindow(object):
    """
    Applies a sliding window to the BOLD time signal and designates a motor task to each window
    :param horizon: length of sliding window/horizon
    :param guard_front: front guard size
    :param guard_back: back guard size
    """

    def __init__(self, params):
        self.horizon = int(params['horizon'])
        self.guard_front = int(params['guard_front'])
        self.guard_back = int(params['guard_back'])

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

        p_new = len(perm)
        assert p_new >= p

        if p_new > p:
            X = self.pad(X)

        for t in range(N):
            # reorder the nodes based on perm order
            X_windowed.append(X[0, perm, t: t + self.horizon])

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
