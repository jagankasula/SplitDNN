import torch
import torch.nn as nn

from torch.nn.modules.container import Sequential
from transformers import BertTokenizer, BertModel

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Model slicing
sub_models=[model]

for sub_model in sub_models:
    all_modules = []

    for module in sub_model.children():
        if isinstance(module,Sequential):
            for m in module.children():
                all_modules.append(m)
        else:
            all_modules.append(module) 


all_modules = nn.Sequential(*all_modules)

all_model = all_modules.eval().to("cpu")

# Example input text
input_text = "Hello, how are you?"

# Tokenize the input text
tokens = tokenizer.tokenize(input_text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_tensor = torch.tensor([input_ids])

print(type(model))
print("---------------------")
print(model)
print("---------------------")
print(type(all_model))
print(all_model)

# Run the input tensor through the model
#outputs = model(input_tensor)
#outputs = all_model(input_tensor)

# Extract the final encoded representation (last layer) of the input sequence
#encoded_input = outputs.last_hidden_state

# Print the encoded representation
#print(encoded_input)
print(model)