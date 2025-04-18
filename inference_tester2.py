#from transformers import GPT2Config 
from modules.custom_models import CustomGPT2LMHeadModel, CustomGPT2Config
from modules.model_mgr import ModelManager
import torch
from modules.utils import load_config
from modules.train_tokenizer import SequenceDataset

# Load config
config = load_config("train_config.yaml")


# Initialize model manager and load model/tokenizer
model_manager = ModelManager(config)
model = model_manager.load_model_from_local()
tokenizer = model_manager.load_fast_tokenizer_from_local()
# model = model_manager.download_model_from_hub()
# tokenizer = model_manager.download_fast_tokenizer_from_hub()
# print("loaded model and tokenizer")
# model_manager.save_model_to_local(model)
# model_manager.save_fast_tokenizer_to_local(tokenizer)

print("shift_method: ", config['pre_processing']['shift_method'])

# test input
print("Test input------------------------")

input = ["+1201+:2000+:32+"]

dataset = SequenceDataset(input, tokenizer,max_length=16, shift_method=config['pre_processing']['shift_method'])

for i in range(len(dataset)):
    print("--------------------------------")
    print(dataset[i])
    input_ids, label_ids, attention_mask = dataset[i]
    input_ids = input_ids.unsqueeze(0)  # Add dimension 0
    label_ids = label_ids.unsqueeze(0)  # Add dimension 0
    attention_mask = attention_mask.unsqueeze(0)  # Add dimension 0
    print("Input IDs : ", input_ids)
    print("Attention Mask : ", attention_mask)
    position_ids = model._get_block_positions(input_ids)
    print("Position IDs : ", position_ids)
    print("Label IDs : ", label_ids)




output = model(
                input_ids,
                attention_mask=attention_mask, 
                position_ids=position_ids,
                shift_labels=label_ids
                )


print("Output shape : ", output.logits.shape)
#print("Output : ", torch.softmax(output.logits[:, -1, :], dim=-1))
probs = torch.softmax(output.logits[:, 5:8, :], dim=-1)
top_probs, top_tokens = torch.topk(probs, k=5)
print(top_probs.shape, top_tokens.shape)
print("Top 5 tokens and probabilities:")
for prob, token in zip(top_probs[0], top_tokens[0]):
    print("--------------------------------")
    for i in range(len(token)):
        print(f"Token: {tokenizer.decode(token[i].item())}, Probability: {prob[i].item():.6f}")

print("Loss : ", output.loss)
