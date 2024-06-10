import os
import yaml


def bool_constructor(loader, node):
    value = loader.construct_scalar(node)
    return value.lower() == 'true'

yaml.add_constructor('tag:yaml.org,2002:bool', bool_constructor, yaml.SafeLoader)

def get_config(dataset_name):

    from dotenv import load_dotenv
    load_dotenv()

    config_path = os.getenv('CONFIG_DIR')
    print(config_path)
    config_file = ''
    if dataset_name.lower() == 'MVTec_AD'.lower():
        config_file = os.path.join(config_path, 'mvtec_ad.yaml')
    elif dataset_name.lower() == 'VisA'.lower():
        config_file = os.path.join(config_path, 'visa.yaml')
    else:
        raise ValueError(f'Invalid dataset name: {dataset_name}')

    with open(config_file) as f:
        config = yaml.safe_load(f)

    return config