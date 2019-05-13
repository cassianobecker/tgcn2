import os
import torch
import configparser

from util.path import get_root
from util.logging import init_loggers

from dataset.hcp.hcp_data import HcpReader, SkipSubjectException
from dataset.hcp.transforms import SlidingWindow, TrivialCoarsening


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


def loaders(device, session, parcellation=None, coarsen=None, batch_size=1):
    """
    Creates train and test datasets and places them in a DataLoader to iterate over.
    """
    train_set = HcpDataset(device, 'train', session, parcellation, coarsen=coarsen)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)

    test_set = HcpDataset(device, 'test', session, parcellation, coarsen=coarsen)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class HcpDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset to host and process the BOLD signal and the associated motor tasks
    """

    def __init__(self, args, device, regime, session, parcellation, coarsen=None):

        settings = get_settings()
        params = get_params()

        init_loggers(settings)

        self.device = device
        self.session = session

        self.reader = HcpReader(settings, params)

        if parcellation is not None:
            self.reader.parcellation = parcellation

        list_url = os.path.join(args.experiment_path, 'conf', regime, session, 'subjects.txt')
        self.subjects = self.reader.load_subject_list(list_url)

        if coarsen is None:
            coarsen = TrivialCoarsening()
        self.coarsen = coarsen

        self.transform = SlidingWindow(params['TIME_SERIES'], coarsen=coarsen)


    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):

        subject = self.subjects[idx]

        try:
            self.reader.logger.info("Feeding subject {:}".format(subject))

            data = self.reader.process_subject(subject, [self.session])

            graph_list, mapping_list = self.coarsen(data['adjacency'])

            graph_list_tensor = self._to_tensor(graph_list)
            mapping_list_tensor = self._to_tensor(mapping_list)

            cues = data['functional'][self.session]['cues']
            ts = data['functional'][self.session]['ts']
            X_windowed, Y_one_hot = self.transform(cues, ts, mapping_list)

        except SkipSubjectException:
            self.reader.logger.warning("Skipping subject {:}".format(subject))

        return X_windowed, Y_one_hot, graph_list_tensor, mapping_list_tensor

    def my_collate(self, batch):
        "Puts each data field into a tensor with outer dimension batch size"
        # batch = filter(lambda x: x is not None, batch)
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    def data_shape(self):
        shape = self.reader.get_adjacency(self.subjects[0]).shape[0]
        return shape

    def self_check(self):
        for subject in self.subjects:
            self.reader.process_subject(subject, [self.session])

    def _to_tensor(self, graph_list):
        coos = [torch.tensor(graph, dtype=torch.long).to(self.device) for graph in graph_list]
        return coos
