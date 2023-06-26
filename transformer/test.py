import pickle

from transformer.modules import Encoder, Decoder, PositionalEncoding

with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

print(loaded_model)