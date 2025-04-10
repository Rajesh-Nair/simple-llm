import torch
from modules.model_mgr import ModelManager
from modules.data_processor import process
from accelerate import Accelerator
import yaml

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from yaml file"""
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

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
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                top_probs, top_tokens = torch.topk(probs, k=5)
                print("Top 5 tokens and probabilities:")
                for prob, token in zip(top_probs[0], top_tokens[0]):
                    print(f"Token: {self.tokenizer.decode(token.item())}, Probability: {prob.item():.4f}")
                

                # Get the new token
                new_token = output[0, -1].item()
                print("All token :", output[0])
                generated.append(new_token)
                
                # Shift everything left and add new token on right
                start_pos = max(0, len(outputs[0]) + 1 - self.config['model']['n_positions'])
                outputs = torch.cat([
                    outputs[:, start_pos:],
                    output[:, -1:] 
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