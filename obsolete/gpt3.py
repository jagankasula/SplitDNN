import torch
import torch.nn as nn
from torch.nn.modules.container import Sequential
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class TransformerModel(nn.Module):
    def __init__(self, model_name):
        super(TransformerModel, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def forward(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        output = self.model.generate(input_ids, max_length=100)
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

# Instantiate the model
model_name = 'gpt2'
model = TransformerModel(model_name)

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
left_modules = list(all_modules.children())[:2]
resnet_left = nn.Sequential(*left_modules)

resnet_left = resnet_left.eval().to("cpu")    
# resnet_left("Jagan")        

# Set device to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Example usage
# input_text = "Hello, how are you?"
# generated_text = model(input_text)
# print(generated_text)

print(model)