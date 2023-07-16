from transformers import AutoConfig, AutoModel, GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
from itertools import islice


# Load the pre-trained GPT-2 model
model = GPT2Model.from_pretrained("gpt2")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

input = tokenizer("Hello, my dog is cute", return_tensors="pt")

# Access the model's components
modules = model._modules  

print(modules)

model_left = dict(islice(modules.items(), 3))

model_left = model_left

model_right = dict(islice(modules.items(), 3, None))

output1 = model_left(**input)

output2 = model_right(*output1)

print(len(modules))
# Split the model into two parts
split_point = 6  # Split after the 6th encoder layer


