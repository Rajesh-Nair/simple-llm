#from transformers import GPT2Config 
from modules.custom_models import CustomGPT2LMHeadModel, CustomGPT2Config
from modules.model_mgr import ModelManager
import torch
import yaml

# load config
def load_config(file_path: str = "config.yaml"):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# train config
train_config = load_config(file_path="train_config.yaml")
config = CustomGPT2Config(**train_config['model'], vocab_size=5)

# config
# Model initialization
# print("1st Stage **************************")
# print("Model initialization------------------------")
# model = CustomGPT2LMHeadModel(config)

# # test input
# print("Test input------------------------")
# input_ids = torch.randint(0, config.vocab_size, (1, 16))
# output = model(input_ids)
# print("Output shape : ", output.logits.shape)

# # Positional Embedding weights requires_grad
# print("Positional Embedding weights requires_grad------------------------")
# if train_config['model']['embedding']['embedding_type'] == 'fixed' or train_config['model']['embedding']['embedding_type'] == 'fixed_block':
#     assert model.transformer.wpe.weight.requires_grad == False
# else:
#     assert model.transformer.wpe.weight.requires_grad == True
# print("Check done------------------------")

# # Padding digit id
# print("Padding digit id------------------------")
# if train_config['model']['embedding']['padding_digit_id'] :
#     assert model.transformer.wpe.padding_idx == train_config['model']['embedding']['padding_digit_id']
# print("Check done------------------------")

# # Save model
# print("Save model------------------------")
# print("Embedding Config : ", model.config.embedding)
# model_mgr = ModelManager(train_config)
# model_mgr.save_model_to_local(model)
# print("Save done------------------------")


# print("2nd Stage **************************")
# # Load model
# print("Load model------------------------")
# model_mgr = ModelManager(train_config)
# model = model_mgr.load_model_from_local()
# print("Load done------------------------")


# # test input
# print("Test input------------------------")
# input_ids = torch.randint(0, config.vocab_size, (1, 16))
# output = model(input_ids)
# print("Output shape : ", output.logits.shape)

# # Positional Embedding weights requires_grad
# print("Positional Embedding weights requires_grad------------------------")
# if train_config['model']['embedding']['embedding_type'] == 'fixed' or train_config['model']['embedding']['embedding_type'] == 'fixed_block':
#     assert model.transformer.wpe.weight.requires_grad == False
# else:
#     assert model.transformer.wpe.weight.requires_grad == True
# print("Check done------------------------")

# # Padding digit id
# print("Padding digit id------------------------")
# if train_config['model']['embedding']['padding_digit_id'] :
#     assert model.transformer.wpe.padding_idx == train_config['model']['embedding']['padding_digit_id']
# print("Check done------------------------")


# # Save model
# print("Save model------------------------")
# model_mgr = ModelManager(train_config)
# model_mgr.upload_model_to_hub(model)
# print("Save done------------------------")


print("3rd Stage **************************")
# Load model
print("Load model------------------------")
model_mgr = ModelManager(train_config)
model = model_mgr.download_model_from_hub()
print("Load done------------------------")


# test input
print("Test input------------------------")
input_ids = torch.randint(0, config.vocab_size, (1, 16))
output = model(input_ids)
print("Output shape : ", output.logits.shape)

# Positional Embedding weights requires_grad
print("Positional Embedding weights requires_grad------------------------")
if train_config['model']['embedding']['embedding_type'] == 'fixed' or train_config['model']['embedding']['embedding_type'] == 'fixed_block':
    assert model.transformer.wpe.weight.requires_grad == False
else:
    assert model.transformer.wpe.weight.requires_grad == True
print("Check done------------------------")

# Padding digit id
print("Padding digit id------------------------")
if train_config['model']['embedding']['padding_digit_id'] :
    assert model.transformer.wpe.padding_idx == train_config['model']['embedding']['padding_digit_id']
print("Check done------------------------")





















