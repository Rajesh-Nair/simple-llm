from transformers import PreTrainedTokenizerFast
from modules.custom_models import CustomGPT2LMHeadModel, CustomGPT2Config
from tokenizers import Tokenizer
import torch
from typing import List, Optional, Tuple
import yaml
import os
from huggingface_hub import login, Repository
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

# Register your custom config and model
CONFIG_MAPPING.register("custom-gpt2", CustomGPT2Config)
MODEL_FOR_CAUSAL_LM_MAPPING.register(CustomGPT2Config, CustomGPT2LMHeadModel)

class ModelManager:
    def __init__(self, config: dict):
        self.config = config
        if self.config['training']['load_checkpoint'] is not None and self.config['training']['load_checkpoint'].startswith("https://huggingface.co"):
            self.hf_config = self.login_to_huggingface()

    # Local paths loaders

    def load_fast_tokenizer_from_local(self) -> PreTrainedTokenizerFast:
        """Load tokenizer from saved file"""
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(self.config['paths']['tokenizer_save_path'], self.config['paths']['tokenizer_file']))
        tokenizer.pad_token = "<|pad|>"
        return tokenizer

    def load_tokenizer_from_local(self) -> Tokenizer:
        """Load tokenizer from saved file"""
        tokenizer = Tokenizer.from_file(os.path.join(self.config['paths']['tokenizer_save_path'], self.config['paths']['tokenizer_file']))
        tokenizer.pad_token = "<|pad|>"
        return tokenizer


    def load_model_from_local(self) -> CustomGPT2LMHeadModel:
        """Load the model from a file"""
        return CustomGPT2LMHeadModel.from_pretrained(self.config['paths']['model_load_path'])
    
    
    def load_checkpoint_from_local(self) -> Tuple[CustomGPT2LMHeadModel, PreTrainedTokenizerFast]:
        """Load the model and tokenizer"""
        model = self.load_model_from_local()
        tokenizer = self.load_fast_tokenizer_from_local()
        tokenizer.pad_token = "<|pad|>"
        return model, tokenizer
    
    # Save to local paths
    
    def save_fast_tokenizer_to_local(self, tokenizer: PreTrainedTokenizerFast):
        """Save the tokenizer to a file"""
        os.makedirs(os.path.dirname(self.config['paths']['tokenizer_save_path']), exist_ok=True)
        tokenizer.save_pretrained(self.config['paths']['tokenizer_save_path'])


    def save_tokenizer_to_local(self, tokenizer: Tokenizer):
        """Save the tokenizer to a file"""
        os.makedirs(os.path.dirname(self.config['paths']['tokenizer_save_path']), exist_ok=True)
        tokenizer.save(os.path.join(self.config['paths']['tokenizer_save_path'], self.config['paths']['tokenizer_file']))
        


    def save_model_to_local(self, model: CustomGPT2LMHeadModel):
        """Save the model to a file"""
        model.save_pretrained(self.config['paths']['model_save_path'])


    # Hugging Face Hub loaders

    def login_to_huggingface(self):
        """Login to Hugging Face Hub"""
        with open("secret.yaml", "r") as f:
            secret_config = yaml.safe_load(f)
            
        hf_config = secret_config["huggingface"]
        login(token=hf_config["token"], add_to_git_credential=True) 
        return hf_config
    

    def download_model_from_hub(self) -> CustomGPT2LMHeadModel:
        # Load model and tokenizer from Hugging Face Hub
        model = CustomGPT2LMHeadModel.from_pretrained(self.config['training']['load_checkpoint'].replace("https://huggingface.co/", ""))
        
        return model
    

    def download_fast_tokenizer_from_hub(self) -> PreTrainedTokenizerFast:
            """Load tokenizer from Hugging Face Hub"""

            # Login to Hugging Face Hub
            hf_config = self.login_to_huggingface()
            
            # Load tokenizer from hub
            fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(
                f"{hf_config['username']}/{hf_config['model_name']}"
            )
            fast_tokenizer.pad_token = "<|pad|>"

            print(f"Successfully loaded fast tokenizer from {hf_config['username']}/{hf_config['model_name']}")
            return fast_tokenizer
    
    def download_repo_from_hub(self):
        """Download entire repository from Hugging Face Hub to local directory"""
        try:
            # Login to Hugging Face Hub
            hf_config = self.login_to_huggingface()
            
            # Create local directory if it doesn't exist
            local_dir = self.config['paths']['model_load_path']
            os.makedirs(local_dir, exist_ok=True)

            # Initialize repository from hub
            repo = Repository(
                local_dir=local_dir,
                clone_from=f"{hf_config['username']}/{hf_config['model_name']}", 
                use_auth_token=hf_config['token'],
                repo_type="model",
                git_user=hf_config["username"],
                git_email=f"{hf_config['username']}@users.noreply.huggingface.co"
            )

            # Pull latest changes, overriding any conflicts
            repo.git_pull(rebase=True)

            print(f"Successfully downloaded repository from {hf_config['username']}/{hf_config['model_name']} to {local_dir}")
            
        except Exception as e:
            print(f"Error downloading repository from Hugging Face Hub: {str(e)}")
            raise

    # Hugging Face Hub uploaders
    
    def upload_model_to_hub(self, model: CustomGPT2LMHeadModel):
        """Upload model to Hugging Face Hub"""
        try:
            # Login to Hugging Face Hub
            hf_config = self.login_to_huggingface()
                        
            # Set up git remote URL with authentication
            remote_url = f"https://{hf_config['name']}:{hf_config['token']}@huggingface.co/{hf_config['username']}/{hf_config['model_name']}"
            
            # Push model to hub using git commands
            model.push_to_hub(
                repo_id=f"{hf_config['username']}/{hf_config['model_name']}",
                commit_message=hf_config["commit_message"],
                use_auth_token=hf_config["token"],
                git_user=hf_config["username"],
                git_email=f"{hf_config['username']}@users.noreply.huggingface.co",
                config={"http.extraheader" : f"AUTHORIZATION: Bearer {hf_config['token']}"}
            )
            print(f"Successfully uploaded model to {hf_config['remote_path']}")
            
        except Exception as e:
            print(f"Error uploading model to Hugging Face Hub: {str(e)}")
            raise

    
    
    def upload_fast_tokenizer_to_hub(self):

        # Login to Hugging Face Hub
        hf_config = self.login_to_huggingface()

        # Convert tokenizer to Hugging Face format 
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(self.config['paths']['tokenizer_save_path'], self.config['paths']['tokenizer_file']))
        

        # Push to hub
        fast_tokenizer.push_to_hub(f"{hf_config['username']}/{hf_config['model_name']}")


    def upload_tokenizer_to_hub(self):
        """Upload tokenizer to Hugging Face Hub"""
        try:
            # Login to Hugging Face Hub
            hf_config = self.login_to_huggingface()
            
            # Get tokenizer file path
            tokenizer_file = os.path.join(self.config['paths']['tokenizer_save_path'], self.config['paths']['tokenizer_file'])
            
            # Upload single file to hub
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=tokenizer_file,
                path_in_repo=self.config['paths']['tokenizer_file'],
                repo_id=f"{hf_config['username']}/{hf_config['model_name']}", 
                token=hf_config['token'],
                repo_type="model"
            )
            
            print(f"Successfully uploaded tokenizer.json to {hf_config['remote_path']}")
            
        except Exception as e:
            print(f"Error uploading tokenizer to Hugging Face Hub: {str(e)}")
            raise 


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
        position_ids = self.model._get_block_positions(input_ids)

        
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
                outputs = torch.cat([
                    outputs[:, 1:],
                    output[:, -1:] 
                ], dim=1)
                
                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask[:, 1:],
                    torch.ones((1,1)).to(self.device)
                ], dim=1)
                
                if self.tokenizer.decode(new_token) == self.config["pre_processing"]["replace_column_delimiter"]:
                    break

        # Decode only the generated tokens
        generated_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return generated_text.replace(" ", "") #self.processor.post_processing(generated_text.replace(" ", ""))