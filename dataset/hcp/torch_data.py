import os
import torch
import configparser
import logging

import numpy as np

from util.path import get_root
import ext.gcn.coarsening as coarsening

from util.encode import one_hot

from dataset.hcp.data import load_subjects, process_subject
from dataset.hcp.downloaders import DtiDownloader, HcpDownloader


def get_settings():
    """
    Creates a ConfigParser object with server/directory/credentials/logging info from preconfigured directory.
    :return: settings, a ConfigParser object
    """
    settings = configparser.ConfigParser()
    settings_dir = os.path.join(get_root(), 'dataset', 'hcp', 'res', 'hcp_database.ini')
    settings.read(settings_dir)
    return settings


def get_params():
    """
    Creates a ConfigParser object with parameter info for reading fMRI data.
    :return: params, a ConfigParser object
    """
    params = configparser.ConfigParser()
    params_dir = os.path.join(get_root(), 'dataset', 'hcp', 'res', 'hcp_experiment.ini')
    params.read(params_dir)
    return params


def set_logger(name, level):
    """
    Creates/retrieves a Logger object with the desired name and level.
    :param name: Name of logger
    :param level: Level of logger
    :return: logger, the configured Logger object
    """
    logger = logging.getLogger(name)
    level_dict = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    logger.setLevel(level_dict[level])
    log_stream = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_stream.setLevel(level_dict[level])
    log_stream.setFormatter(formatter)
    logger.addHandler(log_stream)
    return logger


def loaders(device, batch_size=1):
    """
    Creates train and test datasets and places them in a DataLoader to iterate over.
    :param device: device to send the dataset to
    :param batch_size: fixed at 1 for memory efficiency
    :return: train_loader and test_loader, DataLoaders for the respective datasets
    """
    settings = get_settings()
    params = get_params()

    train_set = HcpDataset(device, settings, params, test=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)

    test_set = HcpDataset(device, settings, params, test=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class HcpDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset to host and process the BOLD signal and the associated motor tasks
    :param device: the device to send the dataset to
    :param settings: ConfigParser that contains server, directory and credential info and logging levels
    :param params: parameters for processing fMRI data
    :param coarsening_levels: level of coarsening to be applied to graph
    :param test: boolean that allows differentiating the train/test urls to pull data from
    """

    def __init__(self, device, settings, params, coarsening_levels=1, test=False):

        self.params = params
        self.settings = settings

        hcp_downloader = HcpDownloader(settings, test)
        dti_downloader = DtiDownloader(settings, test)
        self.loaders = {'hcp_downloader': hcp_downloader, 'dti_downloader': dti_downloader}

        self.list_file = 'subjects.txt'
        if test:
            list_url = os.path.join(get_root(), 'conf/hcp/test/motor_lr', self.list_file)
        else:
            list_url = os.path.join(get_root(), 'conf/hcp/train/motor_lr', self.list_file)

        self.logger = set_logger('HCP_Dataset', settings['LOGGING']['dataloader_logging_level'])

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
    """
    Applies a sliding window to the BOLD time signal and designates a motor task to each window
    :param H: length of sliding window/horizon
    :param Gp: front guard size
    :param Gn: back guard size
    """

    def __init__(self, H, Gp, Gn):
        self.H = H
        self.Gp = Gp
        self.Gn = Gn

    def __call__(self, cues, ts, perm):

        def encode_y(C, X_shape):
            """
            Encodes the target signal to account for windowing
            :param C: targets
            :param X_shape: shape of BOLD data (# examples, # parcels, # time samples)
            :return: y: encoded target signal
            """
            Np, p, T = X_shape
            N = T - self.H + 1

            y = np.zeros([Np, N])
            C_temp = np.zeros(T)
            num_examples = Np * N
            m = C.shape[1]

            for i in range(Np):
                for j in range(m):
                    temp_idx = [idx for idx, e in enumerate(C[i, j, :]) if e == 1]
                    cue_idx1 = [idx - self.Gn for idx in temp_idx]
                    cue_idx2 = [idx + self.Gp for idx in temp_idx]
                    cue_idx = list(zip(cue_idx1, cue_idx2))

                    for idx in cue_idx:
                        C_temp[slice(*idx)] = j + 1

                y[i, :] = C_temp[0: N]

            y = np.reshape(y, num_examples)
            k = np.max(np.unique(y))
            yoh = one_hot(y, k + 1)

            return yoh

        def encode_X(X):
            """
            Provides a list of memory-efficient windowed views into the BOLD time signal.
            :param X: Signal to be encoded
            :return: X_windowed, the list of windowed views
            """

            X_windowed = []
            X = X.astype('float32')
            p, T = X[0].shape
            N = T - self.H + 1

            p_new = len(perm)
            assert p_new >= p

            if p_new > p:
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

        yoh = encode_y(C, X.shape)

        X_windowed = encode_X(X)

        return X_windowed, yoh
