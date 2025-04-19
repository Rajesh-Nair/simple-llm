
"""
Inference Tester Case for Simple LLM

This script provides a test case for the arithmetic capabilities of the trained model.
It demonstrates:
1. How to set up a specific addition problem (101 + 1002 = 1103)
2. How the input is preprocessed with digit reversal (e.g., 101 becomes 101 reversed)
3. How block position IDs (Abacus Embedding) are applied to align digits
4. How the model processes the input to generate predictions
5. How different token prediction rates are handled based on the shift_method configuration

The test case shows the internal representation of the input and expected output,
including tokenization, position IDs, and attention masking, which is useful for
debugging and understanding the model's behavior on arithmetic operations.

The script also supports visualization of:
- Token-by-token processing
- Input/output representation in the model's internal format
- How position encodings align digits for arithmetic operations
- The effect of different shift methods on token prediction

These visualizations help in understanding how the model processes arithmetic
operations and can be used to debug and improve model performance.
"""


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
num1 = 101
num2 = 1002
out = num1 + num2
input = ["+{}+:{}+:{}+".format(str(num1)[::-1], str(num2)[::-1], str(out)[::-1])] # ["+101+:2001+:3011+"]
print("input: ", input)

input_length = len("".join(input[0].split(":")[:-1]))
print("input_length: ", input_length)
dataset = SequenceDataset(input, tokenizer,max_length=16, shift_method=config['pre_processing']['shift_method'])

if config['pre_processing']['shift_method'] == "standard":
    shift_label = 0
elif config['pre_processing']['shift_method'] == "full":
    shift_label = len(input[0].split(":")[-1])
else:
    try:
        shift_label = int(config['pre_processing']['shift_method'])
    except:
        raise ValueError(f"Invalid shift method: {config['pre_processing']['shift_method']}")

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
                labels=label_ids
                )


print("Output shape : ", output.logits.shape)
#print("Output : ", torch.softmax(output.logits[:, -1, :], dim=-1))
probs = torch.softmax(output.logits[:, input_length-1-shift_label:, :], dim=-1)
top_probs, top_tokens = torch.topk(probs, k=5)
print(top_probs.shape, top_tokens.shape)
print("Top 5 tokens and probabilities:")
for prob, token in zip(top_probs[0], top_tokens[0]):
    print("--------------------------------")
    for i in range(len(token)):
        print(f"Token: {tokenizer.decode(token[i].item())}, Probability: {prob[i].item():.6f}")

print("Loss : ", output.loss)

# Extract and visualize attention patterns from each transformer block
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# Create directory for saving attention visualizations
os.makedirs("attention_visualizations", exist_ok=True)

# Get attention outputs from the model
def visualize_attention_patterns(model, input_ids, attention_mask, position_ids):
    # Run the model with output_attentions=True to get attention matrices
    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=True
        )
    
    # outputs.attentions is a tuple of attention tensors for each layer
    # Each tensor has shape (batch_size, num_heads, sequence_length, sequence_length)
    attentions = outputs.attentions
    
    # Create visualizations for each layer and each attention head
    for layer_idx, layer_attention in enumerate(attentions):
        layer_attention = layer_attention.cpu().numpy()
        
        # Get number of attention heads
        num_heads = layer_attention.shape[1]
        
        # Create a figure for this layer with subplots for each head
        fig, axes = plt.subplots(1, num_heads, figsize=(num_heads * 4, 4))
        if num_heads == 1:
            axes = [axes]
        
        # Plot each attention head
        for head_idx, ax in enumerate(axes):
            # Get attention matrix for this head (first item in batch)
            attention_matrix = layer_attention[0, head_idx]
            
            # Create heatmap
            im = ax.imshow(attention_matrix, cmap='viridis')
            ax.set_title(f"Layer {layer_idx+1}, Head {head_idx+1}")
            ax.set_xlabel("Token Position (Key)")
            ax.set_ylabel("Token Position (Query)")
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(f"attention_visualizations/layer_{layer_idx+1}_attention.png")
        plt.close()
        
        print(f"Saved attention visualization for layer {layer_idx+1}")

# Visualize attention patterns for the current input
print("\nGenerating attention visualizations...")
visualize_attention_patterns(model, input_ids, attention_mask, position_ids)
print("Attention visualizations saved to 'attention_visualizations/' directory")

