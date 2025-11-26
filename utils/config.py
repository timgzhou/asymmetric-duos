import yaml
import os
import sys

def load_config(config_path):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"{os.getcwd()=}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
