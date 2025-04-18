# Do not use RopE as this is not fully implemented here

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
import random

# Custom GPT2 Config
class CustomGPT2Config(GPT2Config):
    model_type = "custom-gpt2"

    def __init__(
        self,
        embedding=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding = embedding if embedding is not None else {
            "embedding_type": None
                            }

# Custom GPT2 LM Head Model
class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    config_class = CustomGPT2Config

    def __init__(self, config: GPT2Config, **kwargs):
        super().__init__(config)
        
        # Get embedding parameters from config if available, otherwise from kwargs
        embedding_config = {}

        if hasattr(config, 'embedding'):
            embedding_config = config.embedding 

        self.embedding_type = embedding_config.get('embedding_type', None)
        self.fixed_pos_theta = embedding_config.get('fixed_pos_theta', 10000.0)
        self.fixed_pos_scaling = embedding_config.get('fixed_pos_scaling', 0.1)
        self.fixed_pos_ntk_alpha = embedding_config.get('fixed_pos_ntk_alpha', 1.0)
        self.block_digit_ids = embedding_config.get('block_digit_ids', None)
        self.padding_digit_id = embedding_config.get('padding_digit_id', None)
        self.data_offset = embedding_config.get('data_offset', 0)

            
        self.first_call = False

        # Set padding_idx for both token and position embeddings if padding_digit_id is provided
        if self.padding_digit_id :
            print("**Setting padding_idx for token and position embeddings to ", self.padding_digit_id)
            self.transformer.wte.padding_idx = self.padding_digit_id
            self.transformer.wpe.padding_idx = self.padding_digit_id
        
        if self.embedding_type == 'fixed' or self.embedding_type == 'block_fixed':
            print(f"Embedding type is {self.embedding_type}")

            if self.embedding_type == 'block_fixed':
                assert self.block_digit_ids is not None, "block_digit_ids must be provided for block_fixed embedding"
            
                        
            # Create fixed positional embeddings
            max_positions = config.n_positions
            dim = config.n_embd
            
            # Create position indices
            position = torch.arange(0, max_positions).unsqueeze(1)
            # Create frequency indices
            freq = torch.arange(0, dim // 2).unsqueeze(0)
            
            # Calculate angles with similar formula to RoPE, with scaling and alpha parameters
            theta = self.fixed_pos_theta * (self.fixed_pos_scaling ** 2)
            self.angle = position / (theta ** (2 * freq / dim) * self.fixed_pos_ntk_alpha)
            # Calculate sin and cos values
            pe = torch.zeros(max_positions, dim)
            pe[:, 0::2] = torch.cos(self.angle)
            pe[:, 1::2] = torch.sin(self.angle)
            
            # Replace the learned positional embeddings with fixed ones
            self.transformer.wpe.weight.data.copy_(pe)
            self.transformer.wpe.weight.requires_grad = False


        elif self.embedding_type == 'rope':
            print(f"Embedding type is {self.embedding_type}")
            # Initialize RoPE parameters
            self.rope_theta = self.fixed_pos_theta 
            self.rope_scaling = self.fixed_pos_scaling
            self.rope_ntk_alpha = self.fixed_pos_ntk_alpha
            
            # Replace original positional embeddings with RoPE
            self.transformer.wpe.weight.data.zero_()
            self.transformer.wpe.weight.requires_grad = False
            
            self.rope_cache = {}

        elif self.embedding_type == 'block':
            assert self.block_digit_ids is not None, "block_digit_ids must be provided for block embedding"
            print(f"Embedding type is {self.embedding_type}")
            

        
    
    def _get_block_positions(self, input_ids: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Get block positions for input ids"""
        device = self.device
        self.digits = torch.tensor(self.block_digit_ids, device=device)

        # mask and shape
        mask = torch.isin(input_ids, self.digits)
        mask_shape = mask.shape

        # Create a shifted version of the mask to detect changes from 0 to 1
        shifted_mask = torch.cat([torch.zeros((mask_shape[0], 1), dtype=mask.dtype, device=device), mask[:, :-1]], dim=1)
        starts = (shifted_mask != mask) & mask

        # Generate IDs for each segment of 1s, processing row-wise
        segment_ids = torch.cumsum(starts, dim=1)
        
        # Generate an index array row-wise
        index = torch.arange(mask.size(1), device=device).repeat(mask.size(0), 1)
        
        # Reset index at the start of each segment
        reset_index = torch.zeros_like(mask, device=device).long()
        second_term = index * starts.long()
        reset_index = reset_index.scatter_add(1, segment_ids, second_term)
        
        # Calculate positions in segment
        positions = index - reset_index.gather(1, segment_ids) + 1
        
        # Ensure only values within 1-segments are non-zero
        result = positions * mask

        return result

    
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ) -> torch.Tensor:

        # Get block positions
        if self.embedding_type == 'block_fixed' or self.embedding_type == 'block':
            position_ids = self._get_block_positions(input_ids)

            if self.embedding_type == 'block' and self.data_offset != 0 and self.training:
                device = input_ids.device
                k = random.randint(0, self.data_offset)
                position_ids[position_ids>0] += k
                # Ensure position_ids don't exceed n_positions
                def shift_position_ids(position_ids, max_position):
                    max_allowed = max_position - 1
                    max_pos = position_ids.max()
                    shift = max(0, max_pos - max_allowed)
                    shifted = position_ids - shift
                    # Ensure we don't go below zero
                    shifted = torch.clamp(shifted, 0, max_allowed)
                    return shifted
                position_ids = shift_position_ids(position_ids, self.config.n_positions)

        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


if __name__ == "__main__":
    # Test the custom model
    config = CustomGPT2Config(
        n_positions=16,
        n_embd=256,
        n_layer=8,
        n_head=4,
        vocab_size=5,
        embedding={
            'embedding_type': 'block',
            'fixed_pos_theta': 10000.0,
            'fixed_pos_scaling': 0.1,
            'fixed_pos_ntk_alpha': 1.0,
            'block_digit_ids': [3, 4],
            'padding_digit_id': 1,
            'data_offset': 5
        }
    )
    
    # Create models
    custom_model = CustomGPT2LMHeadModel(config)
    
    # Test input
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Test fixed embedding
    custom_output = custom_model(input_ids)
    
    print("Testing fixed embedding:")
    print(f"Output shape : {custom_output.logits.shape}")
