import json
import os


class dict2obj:
    """Turns dictionary into a class
    """
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])

    def __repr__(self):
        attrs = str([x for x in self.__dict__])
        return '<dict2obj: %s>' % attrs


def read_config(config):
    """Parse a JSON-formatted configuration, either from file or
    as a string. More on the format of configuration file, please
    visit docs/config_format.md

    Args:
      config: either a JSON string or a path to JSON file

    Returs:
      config_dict: config parsed into python's `dict` structure

    Raises:
      ValueError: if the format requirements of the configuration
                  file is not met
    """
    if os.path.isfile(config):
        config = open(config, 'r').read()
    config = json.loads(config)

    required_keys = {'experiment_name': str, 'model': dict, 'metrics': list,
            'feeders': list, 'losses': list}
    for k, t in required_keys.items():
        if not k in config:
            raise ValueError
        if not isinstance(config[k], t):
            raise ValueError

    return config
