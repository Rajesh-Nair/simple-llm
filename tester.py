from transformers import GPT2Config 
from modules.custom_models import CustomGPT2LMHeadModel 
import torch
import numpy as np

# config
config = GPT2Config(n_positions=16, n_embd=256, n_layer=4, n_head=4, vocab_size=50257)
embedding_config = {
    'embedding_type': 'block_fixed',
    'fixed_pos_theta': 10000.0,
    'fixed_pos_scaling': 1.0,
    'fixed_pos_ntk_alpha': 1.0,
    'block_digit_ids': [3,4]
    }
model = CustomGPT2LMHeadModel(config, **embedding_config)

# test input
input_ids = torch.randint(0, config.vocab_size, (1, 16))
output = model(input_ids)
print(output.logits.shape)

# Positional Embedding
print(model.transformer.wpe.weight.shape)

# Positional Embedding weights requires_grad
print(model.transformer.wpe.weight.requires_grad)

# Positional Embedding weights
weights = model.transformer.wpe.weight

# Print weights shape
print(weights.shape)

# Print weights
print(weights)

# Get block positions
input_ids = torch.tensor([[0, 3, 4, 0, 3, 4, 3, 0, 4, 4, 0,3,3,3,3,3]])

weights_post = model.transformer.wpe(input_ids)
print(weights_post)

# Get block positions
weights_post2 = model.get_block_positions(input_ids)
print(weights_post2)

print(model.transformer.wpe(weights_post2))









