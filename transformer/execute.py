import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoderLayer
from torchtext.datasets import WikiText2

from utils import Encoder, Decoder, get_left_modules, get_right_modules, get_test_data, vocab, tokenizer

import torch.utils.data.datapipes.iter.sharding


device = "cpu"
split_point = 2

train_iter = WikiText2(split='train')


ntokens = len(vocab) # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 5 # the number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2 # the number of heads in the Multihead Attention models
dropout = 0.2 # the dropout value


def data_process(raw_text_iter):
  """Converts raw text into a flat Tensor."""
  data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter] # convert to tensor
  """Remove tensors of size 0, convert data to tuple, concatenate tensors along a specified dimension (defualt dimension = 0)."""
  return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def data_process_single_record(raw_text):
  """Converts raw text into a flat Tensor."""
  data = torch.tensor(vocab(tokenizer(raw_text)), dtype=torch.long) # convert to tensor
  """Remove tensors of size 0, convert data to tuple, concatenate tensors along a specified dimension (defualt dimension = 0)."""
  print(f'DATA SIZE:::: *** {data.size()}')
  if data.numel() == 0 or data.size() == (0):
      return None
  else:
    data = torch.cat(tuple(data))
    print(f'TUPLE SIZE:::: *** {len(data)}')
    return data[0]


def execute():
    test_iter = get_test_data()
    print(f'********::: {type(test_iter)}')

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

    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Set model to evaluation mode.
    # model = model.eval().to(device)

    # output = model(test_data[0])

    #print(output)

    # Model splitting.
    left_modules = get_left_modules(split_point, model)
    left_model= left_modules.eval().to(device)

    right_modules = get_right_modules(split_point, model)
    right_model = right_modules.eval().to(device)

    for item in test_iter:
        print(item)
        # train_data = data_process(train_iter)
        # val_data = data_process(val_iter)
        
        test_data = data_process_single_record(item)

        if test_data != None:

            # Running left model.
            left_output = left_model(test_data[0])

            # Running right model.
            output = right_model(left_output)

            print(output)

    with open('model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    print(loaded_model)


execute()