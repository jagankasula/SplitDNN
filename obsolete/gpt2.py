from __future__ import annotations

import torch

from transformers import AutoConfig, AutoModel, GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
from torch.nn.modules.container import Sequential
from flops_profiler.profiler import get_model_profile
from utils import get_left_model

name = 'gpt2'
use_cuda = True
device = torch.device('cuda:0') if torch.cuda.is_available(
) and use_cuda else torch.device('cpu')
tokenizer = GPT2Tokenizer.from_pretrained(name)
model = GPT2Model.from_pretrained(name)
#model = model.to(device)

left_model = get_left_model(3, model)
left_model.eval().to(device)
 


batch_size = 1
seq_len = 128
input = tokenizer("Hello, my dog is cute", return_tensors="pt")

print(f'****** {type(input)}')

outputs = model(**input)

last_hidden_states = outputs.last_hidden_state

print(last_hidden_states)

# flops, macs, params = get_model_profile(
#     model,
#     kwargs=input,
#     print_profile=True,
#     detailed=True,
# )

# utils.print_output(flops, macs, params)