import yaml

def load_yaml_config(file_path):
    with open(file_path, "r") as infile:
        configs = yaml.safe_load(infile)
    return configs
