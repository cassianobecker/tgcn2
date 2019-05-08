import logging
import os
import sys

import nibabel as nib

from util.path import get_root

LOG_FILE = os.path.join(get_root(), 'logger.log')
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=LOG_FILE)


def set_logger(name, level):
    """
    Creates/retrieves a Logger object with the desired name and level.
    :param name: Name of logger
    :param level: Level of logger
    :return: logger, the configured Logger object
    """
    logger = logging.getLogger(name)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    level_dict = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    logger.setLevel(level_dict[level])
    return logger


def get_logger(name):
    return logging.getLogger(name)


def init_loggers(settings):
    set_logger('HcpDownloader', settings['LOGGING']['downloader_logging_level'])
    set_logger('HcpDataset', settings['LOGGING']['dataloader_logging_level'])
    set_logger('DtiDownloader', settings['LOGGING']['downloader_logging_level'])
    nib.imageglobals.logger = set_logger('Nibabel', settings['LOGGING']['nibabel_logging_level'])
