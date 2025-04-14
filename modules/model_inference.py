import torch
from modules.model_mgr import ModelManager
from modules.data_processor import process
from accelerate import Accelerator
from modules.utils import load_config



class TextGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.model_manager = ModelManager(config)
        self.model, self.tokenizer = self.model_manager.load_checkpoint_from_local()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = process(config)

    def generate_tokens(self,prompt:str,max_length:int, tokens_rate: int = 1, verbose: bool = False):
        """Generate tokens using the loaded model"""
        if max_length is None:
            max_length = self.config['model']['n_positions']

        # Preprocess and encode prompt
        processed_prompt = prompt #self.processor.pre_processing(" "+prompt.strip()+" ")
        input_ids = self.tokenizer.encode(processed_prompt)
        
        # Convert to tensor and move to device
        input_tensor = torch.tensor([input_ids]).to(self.device)
        attention_mask = torch.tensor([[1] * len(input_ids)]).to(self.device)
        position_ids = self.model._get_block_positions(input_tensor)

        
        with torch.no_grad():
            outputs = input_tensor
            generated = []

            for _ in range(max_length):
                # Get model predictions
                logits = self.model(outputs, attention_mask=attention_mask, position_ids=position_ids).logits
                if verbose:
                    print("Logits shape:", logits.shape)
                
                # Get probabilities for the last tokens_rate positions
                # This handles models that can predict multiple tokens at once
                next_token_logits = logits[:, -tokens_rate:, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # Get top predictions for each position in the tokens_rate window
                top_probs, top_tokens = torch.topk(probs, k=5)
                
                # Print top tokens and probabilities for the last position
                if verbose:
                    print("Top 5 tokens and probabilities:")
                    for batch_idx in range(top_probs.shape[0]):
                        for pos_idx in range(top_probs.shape[1]):
                            for i in range(5):  # k=5
                                token = top_tokens[batch_idx, pos_idx, i].item()
                                prob = top_probs[batch_idx, pos_idx, i].item()
                                print(f"Token: {self.tokenizer.decode(token)}, Probability: {prob:.4f}")
                
                # Select the most likely token for each position
                new_tokens = torch.argmax(probs, dim=-1)  # Shape: [batch_size, tokens_rate]
                
                # Process each new token and check stopping conditions
                should_stop = False
                for i in range(new_tokens.shape[1]):
                    token = new_tokens[0, i].item()
                    generated.append(token)
                    
                    # Check stopping conditions for each token
                    if self.tokenizer.decode(token) == self.config["pre_processing"]["replace_column_delimiter"]:
                        if verbose:
                            print("Column delimiter detected, stopping generation")
                        should_stop = True
                        break
                    
                    if token == self.tokenizer.pad_token_id:
                        if verbose:
                            print("Pad token id detected, stopping generation")
                        should_stop = True
                        break
                
                # Append new tokens to outputs
                outputs = torch.cat([outputs, new_tokens], dim=1)

                # Break out of the main generation loop if stopping condition was met
                if should_stop:
                    break
                
                # Update attention mask
                new_attention = torch.ones((outputs.shape[0], tokens_rate), device=self.device)
                attention_mask = torch.cat([attention_mask, new_attention], dim=1)
                
                # Update position IDs for the new tokens
                position_ids = self.model._get_block_positions(outputs)
                


        # Decode only the generated tokens
        generated_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return generated_text.replace(" ", "") #self.processor.post_processing(generated_text.replace(" ", ""))

    def generate_text(self, prompt: str, max_length: int = None) -> str:
        """Generate text using the loaded model"""
        if max_length is None:
            max_length = self.config['model']['n_positions']

        # Preprocess and encode prompt
        processed_prompt = prompt #self.processor.pre_processing(" "+prompt.strip()+" ")
        input_ids = self.tokenizer.encode(processed_prompt)
        
        
        # Pad to model position length, aligning tokens to right
        #pad_length = self.config['model']['n_positions'] - len(input_ids)
        #padded_input = [self.tokenizer.pad_token_id] * pad_length + input_ids

        # Convert to tensor and move to device
        input_tensor = torch.tensor([input_ids]).to(self.device)
        attention_mask = torch.tensor([[1] * len(input_ids)]).to(self.device)
        position_ids = self.model._get_block_positions(input_tensor)

        
        with torch.no_grad():
            outputs = input_tensor
            generated = []

            for _ in range(max_length):

                # Generate next token
                output = self.model.generate(
                    outputs,
                    max_length=outputs.size(1)+1,
                    attention_mask=attention_mask, 
                    position_ids=position_ids,
                    bos_token_id=None,
                    eos_token_id=None,
                    do_sample=True,
                    top_k=1,
                    top_p=0.95,
                    temperature=0.01
                )
                
                print("Input : ", self.tokenizer.decode(outputs[0],skip_special_tokens=True), \
                      "\nOutput : ", self.tokenizer.decode(output[0],skip_special_tokens=True ))
                # Get probabilities for the next token
                logits = self.model(outputs, attention_mask=attention_mask, position_ids=position_ids).logits
                print("Logits : ", logits.shape)
                look_at = -1
                probs = torch.softmax(logits[:, look_at, :], dim=-1)
                top_probs, top_tokens = torch.topk(probs, k=5)
                print("Top 5 tokens and probabilities:")
                for prob, token in zip(top_probs[0], top_tokens[0]):
                    print(f"Token: {self.tokenizer.decode(token.item())}, Probability: {prob.item():.4f}")
                

                # Get the new token
                new_token = output[0, look_at].item()
                print("All token :", output[0])
                generated.append(new_token)
                
                # Shift everything left and add new token on right
                start_pos = max(0, len(outputs[0]) + 1 - self.config['model']['n_positions'])
                outputs = torch.cat([
                    outputs[:, start_pos:],
                    output[:, look_at:] 
                ], dim=1)
                
                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask[:, start_pos:],
                    torch.ones((1,1)).to(self.device)
                ], dim=1)
                
                if self.tokenizer.decode(new_token) == self.config["pre_processing"]["replace_column_delimiter"] :
                    print("Column delimiter detected, stopping generation")
                    break
            
                if new_token == self.tokenizer.pad_token_id :
                    print("Pad token id detected, stopping generation")
                    break

        # Decode only the generated tokens
        generated_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return generated_text.replace(" ", "") #self.processor.post_processing(generated_text.replace(" ", ""))