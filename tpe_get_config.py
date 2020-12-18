# -*- coding: utf-8 -*-

import configparser
import os

def getConfig(section, key):
    """Read configuration file.

    Parse configuration file with configparser module to get values.

    Args:
        section: configuration sections.
        key: configuration keys.

    Returns:
        Value for given section and key.
    """
    
    config = configparser.ConfigParser()
    path = os.path.split(os.path.realpath(__file__))[0] + '/tpe.conf'
    config.read(path)
    return config.get(section, key)