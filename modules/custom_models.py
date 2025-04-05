# Do not use RopE as this is not fully implemented here

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
import math

class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config: GPT2Config, embedding_type: str = None, **kwargs):
        super().__init__(config)
        self.embedding_type = embedding_type
        
        if embedding_type == 'rope':
            # Initialize RoPE parameters
            self.rope_theta = kwargs.get('rope_theta', 10000.0)
            self.rope_scaling = kwargs.get('rope_scaling', 1.0)
            self.rope_ntk_alpha = kwargs.get('rope_ntk_alpha', 1.0)
            
            # Replace original positional embeddings with RoPE
            self.transformer.wpe.weight.data.zero_()
            self.transformer.wpe.weight.requires_grad = False
            
            self.rope_cache = {}
        
        elif embedding_type == 'fixed':
            # Initialize fixed positional embedding parameters
            self.fixed_pos_theta = kwargs.get('fixed_pos_theta', 10000.0)
            self.fixed_pos_scaling = kwargs.get('fixed_pos_scaling', 1.0)
            self.fixed_pos_ntk_alpha = kwargs.get('fixed_pos_ntk_alpha', 1.0)
            
            # Create fixed positional embeddings
            max_positions = config.n_positions
            dim = config.n_embd
            
            # Create position indices
            position = torch.arange(max_positions).unsqueeze(1)
            # Create frequency indices
            freq = torch.arange(dim // 2).unsqueeze(0)
            
            # Calculate angles with similar formula to RoPE, with scaling and alpha parameters
            theta = self.fixed_pos_theta * (self.fixed_pos_scaling ** 2)
            angle = position / (theta ** (2 * freq / dim) * self.fixed_pos_ntk_alpha)
            print(angle.shape)
            # Calculate sin and cos values
            pe = torch.zeros(max_positions, dim)
            pe[:, 0::2] = torch.sin(angle)
            pe[:, 1::2] = torch.cos(angle)
            
            # Replace the learned positional embeddings with fixed ones
            self.transformer.wpe.weight.data.copy_(pe)
            self.transformer.wpe.weight.requires_grad = False
            
    def _get_rope_cache(self, seq_len: int, device: torch.device) -> tuple:
        """Cache RoPE embeddings for efficiency"""
        if seq_len not in self.rope_cache:
            # Calculate RoPE embeddings
            dim = self.config.n_embd // self.config.n_head
            theta = self.rope_theta * (self.rope_scaling ** 2)
            
            # Create position indices
            pos = torch.arange(seq_len, device=device).unsqueeze(1)
            # Create frequency indices
            freq = torch.arange(dim // 2, device=device).unsqueeze(0)
            
            # Calculate RoPE angles with NTK scaling
            angle = pos / (theta ** (2 * freq / dim) * self.rope_ntk_alpha)
            
            # Calculate sin and cos values
            sin = torch.sin(angle)
            cos = torch.cos(angle)

            # Double the dimension by repeating each value
            sin = sin.repeat_interleave(2, dim=-1)
            cos = cos.repeat_interleave(2, dim=-1)
            
            # Cache the results
            self.rope_cache[seq_len] = (sin, cos)
            
        return self.rope_cache[seq_len]
    
    def _apply_rope(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to input tensor"""
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Get cached RoPE values
        sin, cos = self._get_rope_cache(seq_len, device)
        
        # Reshape for RoPE application
        x_reshaped = x.view(batch_size, seq_len, self.config.n_head, -1)
        
        # Split into real and imaginary parts
        x_cos = x_reshaped
        x_sin = torch.roll(x_reshaped.clone(), shifts=1, dims=-1)
        x_sin[..., 0::2] = torch.roll(x_reshaped.clone(), shifts=-1, dims=-1)[..., 0::2]*-1  # Negate every other element in the last dimension
        
        
        # Reshape sin and cos for broadcasting
        sin = sin.unsqueeze(0).unsqueeze(2)  # Add batch and head dimensions
        cos = cos.unsqueeze(0).unsqueeze(2)  # Add batch and head dimensions
        
        # Apply RoPE rotation
        x_rope = x_cos * cos + x_sin * sin
        
        return x_rope.view(batch_size, seq_len, -1)
    
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
    ) -> torch.Tensor:
        if self.embedding_type == 'rope':
            # Get the embeddings and attention outputs from parent model
            outputs = super().forward(
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
            
            # Extract and modify attention layers to apply RoPE to Q and K vectors
            for layer in self.transformer.h:
                # Get query and key from attention layer
                query = layer.attn.q_proj(outputs.hidden_states)
                key = layer.attn.k_proj(outputs.hidden_states)
                
                # Apply RoPE to query and key
                query_rope = self._apply_rope(query, position_ids)
                key_rope = self._apply_rope(key, position_ids)
                
                # Replace the original query and key with RoPE versions
                layer.attn.q_proj = lambda x: query_rope
                layer.attn.k_proj = lambda x: key_rope
            
            return outputs
        else:
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
    config = GPT2Config(
        n_positions=16,
        n_embd=256,
        n_layer=4,
        n_head=4,
        vocab_size=50257
    )
    
    # Create models
    original_model = GPT2LMHeadModel(config)
    custom_model = CustomGPT2LMHeadModel(config, embedding_type=None)
    custom_model_rope = CustomGPT2LMHeadModel(config, embedding_type='rope')
    custom_model_fixed_pos = CustomGPT2LMHeadModel(config, embedding_type='fixed')
    
    # Copy weights
    custom_model.load_state_dict(original_model.state_dict())
    
    # Test input
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Test without RoPE
    original_output = original_model(input_ids)
    custom_output = custom_model(input_ids)
    
    print("Testing without RoPE:")
    print(f"Output shapes match: {original_output.logits.shape == custom_output.logits.shape}")
    
    # Test with RoPE
    rope_output = custom_model_rope(input_ids)
    print("\nTesting with RoPE:")
    print(f"Output shape: {rope_output.logits.shape}")
    
    # Test with fixed positional embeddings
    fixed_pos_output = custom_model_fixed_pos(input_ids)
    print("\nTesting with fixed positional embeddings:")
    print(f"Output shape: {fixed_pos_output.logits.shape}")
