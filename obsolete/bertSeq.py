import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class BertSequential(nn.Sequential):
    def __init__(self):
        super(BertSequential, self).__init__(
            nn.Embedding(30522, 768, padding_idx=0),
            nn.Embedding(512, 768),
            nn.Embedding(2, 768),
            nn.LayerNorm(768, eps=1e-12, elementwise_affine=True),
            nn.Dropout(p=0.1, inplace=False),
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(768, 768, bias=True),
                    nn.Linear(768, 768, bias=True),
                    nn.Linear(768, 768, bias=True),
                    nn.Dropout(p=0.1, inplace=False),
                    nn.Linear(768, 768, bias=True),
                    nn.LayerNorm(768, eps=1e-12, elementwise_affine=True),
                    nn.Dropout(p=0.1, inplace=False),
                    nn.Linear(768, 3072, bias=True),
                    nn.GELU(),  # Assuming GELUActivation is equivalent to GELU
                    nn.Linear(3072, 768, bias=True),
                    nn.LayerNorm(768, eps=1e-12, elementwise_affine=True),
                    nn.Dropout(p=0.1, inplace=False)
                )
                for _ in range(12)
            ]),
            nn.Linear(768, 768, bias=True),
            nn.Tanh()
        )

model = BertSequential()
model = model.eval().to("cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example input text
input_text = "Hello, how are you?"

# Tokenize the input text
tokens = tokenizer.tokenize(input_text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_tensor = torch.tensor([input_ids])

outputs = model(input_tensor)

# Extract the final encoded representation (last layer) of the input sequence
encoded_input = outputs.last_hidden_state

# Print the encoded representation
print(encoded_input)
