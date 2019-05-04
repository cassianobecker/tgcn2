import os
import torch
import configparser

import numpy as np

from util.path import get_root
import ext.gcn.coarsening as coarsening

from util.encode import one_hot
from util.logging import set_logger

from dataset.hcp.data import load_subjects, process_subject
from dataset.hcp.downloaders import DtiDownloader, HcpDownloader


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


def loaders(device, batch_size=1, download_train=True, download_test=True):
    """
    Creates train and test datasets and places them in a DataLoader to iterate over.
    :param device: device to send the dataset to
    :param batch_size: fixed at 1 for memory efficiency
    :param download_train: whether to attempt downloading training data, set to False if data available locally
    :param download_test: whether to attempt downloading test data, set to False if data available locally
    :return: train_loader and test_loader, DataLoaders for the respective datasets
    """
    settings = get_settings()
    params = get_params()

    train_set = HcpDataset(device, settings, params, test=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)

    test_set = HcpDataset(device, settings, params, test=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if download_train:
        train_set.download_all()
    if download_test:
        test_set.download_all()

    return train_loader, test_loader, train_set.infer_size()


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

        self.hcp_downloader = HcpDownloader(settings, test)
        self.dti_downloader = DtiDownloader(settings, test)
        self.loaders = {'hcp_downloader': self.hcp_downloader, 'dti_downloader': self.dti_downloader}

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

    def infer_size(self):
        subject = self.subjects[0]

        data = process_subject(self.params, subject, [self.session], self.loaders)
        size = data['adj'].shape[0]
        return size

    def download_all(self):
        for subject in self.subjects:

            for task in [self.session]:
                fname = 'tfMRI_' + task + '_Atlas.dtseries.nii'
                furl = os.path.join('HCP_1200', subject, 'MNINonLinear', 'Results', 'tfMRI_' + task, fname)
                self.hcp_downloader.load(furl)

                furl = os.path.join('HCP_1200', subject, 'MNINonLinear', 'Results', 'tfMRI_' + task, 'EVs')
                files = ['cue.txt', 'lf.txt', 'lh.txt', 'rf.txt', 'rh.txt', 't.txt']

                for file in files:
                    new_path = os.path.join(furl, file)
                    self.hcp_downloader.load(new_path)

                parc = self.params['PARCELLATION']['parcellation']
                fpath = os.path.join('HCP_1200', subject, 'MNINonLinear', 'fsaverage_LR32k')
                suffixes = {'aparc': '.aparc.a2009s.32k_fs_LR.dlabel.nii',
                            'dense': '.aparc.a2009s.32k_fs_LR.dlabel.nii'}
                parc_furl = os.path.join(fpath, subject + suffixes[parc])
                self.hcp_downloader.load(parc_furl)

                fname = 'tfMRI_' + task + '_Physio_log.txt'
                furl = os.path.join('HCP_1200', subject, 'MNINonLinear', 'Results', 'tfMRI_' + task, fname)
                self.hcp_downloader.load(furl)

                inflation = 'inflated'
                hemis = ['L', 'R']
                for hemi in hemis:
                    fname = subject + '.' + hemi + '.' + inflation + '.32k_fs_LR.surf.gii'
                    furl = os.path.join('HCP_1200', subject, 'MNINonLinear', 'fsaverage_LR32k', fname)
                    self.hcp_downloader.load(furl)

                furl = os.path.join('HCP_1200', subject, 'MNINonLinear', 'Results', 'dMRI_CONN')
                file = furl + '/' + subject + '.aparc.a2009s.dti.conn.mat'

                if subject in self.dti_downloader.whitelist:
                    self.dti_downloader.load(file)

        self.logger.info("Downloading All patient data: Completed")


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
                    temp_idx = [idx for idx, e in enumerate(C[i, j, :]) if e == 1]      # find indices in the original signal with the task
                    cue_idx1 = [idx - self.Gn for idx in temp_idx]      # starting indices of the task in the new signal (calculated with guards)
                    cue_idx2 = [idx + self.Gp for idx in temp_idx]      # ending indices of the task in the new signal (calculated with guards)
                    cue_idx = list(zip(cue_idx1, cue_idx2))             # pair the tuples to form intervals to assign to specific motor task

                    for idx in cue_idx:
                        C_temp[slice(*idx)] = j + 1                     # assign task to specified interval

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

            X_windowed = []     # we store the views into the signal in a list
            X = X.astype('float32')
            p, T = X[0].shape
            N = T - self.H + 1      # resulting time signal will have (time_dimension - window_length + 1) time steps

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
