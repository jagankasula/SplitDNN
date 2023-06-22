import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoderLayer
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from utils import get_left_modules, get_right_modules


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
    

device = "cpu"
split_point = 2

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


ntokens = len(vocab) # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 6 # the number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2 # the number of heads in the Multihead Attention models
dropout = 0.2 # the dropout value


def data_process(raw_text_iter):
  """Converts raw text into a flat Tensor."""
  data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter] # convert to tensor
  """Remove tensors of size 0, convert data to tuple, concatenate tensors along a specified dimension (defualt dimension = 0)."""
  return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_iter, val_iter, test_iter = WikiText2()
print(type(test_iter))
# train_data = data_process(train_iter)
# val_data = data_process(val_iter)
test_data = data_process(test_iter)
print(type(test_data))

# Add encoder in the beginning.
tmp_list = [Encoder(ntokens, emsize, dropout)]
module_list = []
module_list.append(nn.Sequential(*tmp_list))
tmp_list = []

# Add all the necessary transformer blocks.
for i in range(nlayers):
    transformer_block = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
    tmp_list.append(transformer_block.to(device))


# Add decoder in the end.
tmp_list.append(Decoder(ntokens, emsize))
module_list.append(nn.Sequential(*tmp_list))


model = torch.nn.Sequential(*module_list)

# Set model to evaluation mode.
model = model.eval().to(device)

print(len(test_data))

# output = model(test_data[0])

#print(output)

# Model splitting.
left_modules = get_left_modules(split_point, model)
left_model= left_modules.eval().to(device)

right_modules = get_right_modules(split_point, model)
right_model = right_modules.eval().to(device)

# Running left model.
left_output = left_model(test_data[0])

# Running right model.
output = right_model(left_output)

print(output)

# print(all_model)