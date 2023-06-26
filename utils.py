import csv
import cv2
import datetime
import json
import numpy as np
import pickle
import ray
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from json import JSONEncoder
from torch.nn.modules.container import Sequential

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
    
def get_left_modules(split_point, model):
    all_modules = get_all_modules(model)
    all_modules = nn.Sequential(*all_modules)
    left_modules = list(all_modules.children())[:split_point]
    print(f'No. left modules: {len(left_modules)}')
    left_modules = nn.Sequential(*left_modules)
    return left_modules

def get_right_modules(split_point, model):
    all_modules = get_all_modules(model)
    all_modules = nn.Sequential(*all_modules)
    right_modules = list(all_modules.children())[split_point:]
    print(f'No. right modules: {len(right_modules)}')
    right_modules = nn.Sequential(*right_modules)
    return right_modules

def get_all_modules(model):
    all_modules = []

    for module in model.children():
        if isinstance(module,Sequential):
            for m in module.children():
                all_modules.append(m)
        else:
            all_modules.append(module) 


    all_modules = nn.Sequential(*all_modules)
    return all_modules


def create_test_tokens(
    batch_size, seq_len, device=torch.device('cpu'), has_token_type_ids=False,
):
    tokens = {}
    mask_lens = torch.randint(low=1, high=seq_len, size=(batch_size,))

    mask = (
        torch.arange(seq_len).expand(batch_size, seq_len)
        < mask_lens.unsqueeze(1)
    ).to(device)
    tokens['attention_mask'] = mask.to(torch.int64)

    tokens['input_ids'] = torch.randint(
        low=1000, high=10000, size=(batch_size, seq_len), dtype=torch.int64,
    ).to(device)

    tokens['input_ids'] = tokens['input_ids'].masked_fill(~mask, 0)
    for i in range(batch_size):
        tokens['input_ids'][i, 0] = 101
        tokens['input_ids'][i, mask_lens[i] - 1] = 102

    if has_token_type_ids:
        tokens['token_type_ids'] = torch.LongTensor(
            [[0] * seq_len] * batch_size,
        ).to(device)
    return tokens    

def convert_image_to_tensor(img, device):

    img_rgb = Image.fromarray(img).convert('RGB')
    resize = transforms.Resize([224, 224])
    img_rgb = resize(img_rgb)

    to_tensor = transforms.ToTensor()
    tensor = to_tensor(img_rgb)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)

    return tensor

def get_video_input(file_path):
    # Read the input from the file.
    cam = cv2.VideoCapture('hdvideo.mp4')
    return cam

def print_output(flops, macs, params):
    print('{:<30}  {:<8}'.format('Number of flops: ', flops))
    print('{:<30}  {:<8}'.format('Number of MACs: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def write_to_csv(filename, field_names, data):
    # Check if the file exists
    file_exists = False
    try:
        with open(filename, 'r') as file:
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    # Open the CSV file in the appropriate mode
    mode = 'a' if file_exists else 'w'
    with open(filename, mode, newline='') as file:
        writer = csv.writer(file)

        # Write a new line if the file is empty
        if not file_exists:
            writer.writerow(field_names)  # Example column headers

        # Write the data to the file
        writer.writerow(data)


# Process video
def video_process(model):
    count = 0
    cam = get_video_input('hdvideo.mp4')

    while count < 10:
        ret, img = cam.read()
        tensor = torch.tensor(img, dtype=torch.long)
        count += 1
        output = model(tensor.view(-1, tensor.size(-1)))
        print(output)