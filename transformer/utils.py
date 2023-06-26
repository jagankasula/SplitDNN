import datetime
import json
import math
import numpy as np
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from torch.nn.modules.container import Sequential
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer as torch_tokenizer
from torchtext.vocab import build_vocab_from_iterator


def get_tokenizer():
    return torch_tokenizer('basic_english')

tokenizer = get_tokenizer()

def get_vocab():
    train_iter = WikiText2(split='train')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

vocab = get_vocab()

class Encoder(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(Encoder, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # Need (S, N) format for encoder.
        src = src.t()
        src = self.encoder(src) * math.sqrt(self.ninp)
        return self.pos_encoder(src)

class Decoder(nn.Module):
    def __init__(self, ntoken, ninp):
        super(Decoder, self).__init__()
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp):
        # Need batch dimension first for output of pipeline.
        return self.decoder(inp).permute(1, 0, 2)
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Config:
    @staticmethod
    def get_config():
        with open('config.json') as config_file:
            config = json.load(config_file)
            config['url'] = config['url'].replace("{{server}}", config['server'])
        return config
    
    
class Logger():
    def log(message):
        time = datetime.datetime.now()
        print(f"{time}::{message}")
    

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

def get_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def get_test_data():
    train_raw, val_raw, test_raw = WikiText2()
    # train_data = data_process(train_raw)
    # val_data = data_process(val_raw)
    return test_raw

def print_output(flops, macs, params):
    print('{:<30}  {:<8}'.format('Number of flops: ', flops))
    print('{:<30}  {:<8}'.format('Number of MACs: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
