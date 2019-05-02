import logging


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