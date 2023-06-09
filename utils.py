import datetime
import json
import numpy as np
import torch.nn as nn
import torchvision.models as models

from json import JSONEncoder

my_models = {
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101
}

class Config:
    @staticmethod
    def get_config():
        with open('config.json') as config_file:
            config = json.load(config_file)
            config['url'] = config['url'].replace("{{server}}", config['server'])
        return config
    
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
class Logger():
    def log(message):
        time = datetime.datetime.now()
        print(f"{time}::{message}")

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


    