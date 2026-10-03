import yaml
import json
def read_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def read_json(json_path:str):
    _json = None
    with open(json_path, "r") as f:
        _json = json.load(f)
    return _json