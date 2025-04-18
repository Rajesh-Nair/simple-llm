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
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(self.config['paths']['model_local_path'], self.config['paths']['model_name'], self.config['paths']['tokenizer_name']))
        tokenizer.pad_token = "<|pad|>"
        return tokenizer

    def load_tokenizer_from_local(self) -> Tokenizer:
        """Load tokenizer from saved file"""
        tokenizer = Tokenizer.from_file(os.path.join(self.config['paths']['model_local_path'], self.config['paths']['model_name'], self.config['paths']['tokenizer_name']))
        tokenizer.pad_token = "<|pad|>"
        return tokenizer


    def load_model_from_local(self) -> CustomGPT2LMHeadModel:
        """Load the model from a file"""
        return CustomGPT2LMHeadModel.from_pretrained(os.path.join(self.config['paths']['model_local_path'], self.config['paths']['model_name']))
    
    
    def load_checkpoint_from_local(self) -> Tuple[CustomGPT2LMHeadModel, PreTrainedTokenizerFast]:
        """Load the model and tokenizer"""
        model = self.load_model_from_local()
        tokenizer = self.load_fast_tokenizer_from_local()
        tokenizer.pad_token = "<|pad|>"
        return model, tokenizer
    
    # Save to local paths
    
    def save_fast_tokenizer_to_local(self, tokenizer: PreTrainedTokenizerFast):
        """Save the tokenizer to a file"""
        os.makedirs(os.path.dirname(os.path.join(self.config['paths']['model_local_path'], self.config['paths']['model_name'])), exist_ok=True)
        tokenizer.save_pretrained(os.path.join(self.config['paths']['model_local_path'], self.config['paths']['model_name']))


    def save_tokenizer_to_local(self, tokenizer: Tokenizer):
        """Save the tokenizer to a file"""
        os.makedirs(os.path.dirname(os.path.join(self.config['paths']['model_local_path'], self.config['paths']['model_name'])), exist_ok=True)
        tokenizer.save_pretrained(os.path.join(self.config['paths']['model_local_path'], self.config['paths']['model_name']))


    def save_model_to_local(self, model: CustomGPT2LMHeadModel):
        """Save the model to a file"""
        model.save_pretrained(os.path.join(self.config['paths']['model_local_path'], self.config['paths']['model_name']))


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
                f"{hf_config['username']}/{self.config['paths']['model_name']}"
            )
            fast_tokenizer.pad_token = "<|pad|>"

            print(f"Successfully loaded fast tokenizer from {hf_config['username']}/{self.config['paths']['model_name']}")
            return fast_tokenizer
    
    def download_repo_from_hub(self):
        """Download entire repository from Hugging Face Hub to local directory"""
        try:
            # Login to Hugging Face Hub
            hf_config = self.login_to_huggingface()
            
            # Create local directory if it doesn't exist
            local_dir = os.path.join(self.config['paths']['model_local_path'], self.config['paths']['model_name'])
            os.makedirs(local_dir, exist_ok=True)

            # Initialize repository from hub
            repo = Repository(
                local_dir=local_dir,
                clone_from=self.config['training']['load_checkpoint'].replace("https://huggingface.co/", ""), #f"{hf_config['username']}/{hf_config['model_name']}", 
                use_auth_token=hf_config['token'],
                repo_type="model",
                git_user=hf_config["username"],
                git_email=f"{hf_config['username']}@users.noreply.huggingface.co"
            )

            # Pull latest changes, overriding any conflicts
            repo.git_pull(rebase=True)

            print(f"Successfully downloaded repository from {self.config['training']['load_checkpoint']} to {local_dir}")
            
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
            remote_url = f"https://{hf_config['name']}:{hf_config['token']}@huggingface.co/{hf_config['username']}/{self.config['paths']['model_name']}"
            
            # Push model to hub using git commands
            model.push_to_hub(
                repo_id=f"{hf_config['username']}/{self.config['paths']['model_name']}",
                commit_message=hf_config["commit_message"],
                use_auth_token=hf_config["token"],
                git_user=hf_config["username"],
                git_email=f"{hf_config['username']}@users.noreply.huggingface.co",
                config={"http.extraheader" : f"AUTHORIZATION: Bearer {hf_config['token']}"}
            )
            print(f"Successfully uploaded model to {hf_config['username']}/{self.config['paths']['model_name']}")
            
        except Exception as e:
            print(f"Error uploading model to Hugging Face Hub: {str(e)}")
            raise

    
    
    def upload_fast_tokenizer_to_hub(self):

        # Login to Hugging Face Hub
        hf_config = self.login_to_huggingface()

        # Convert tokenizer to Hugging Face format 
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(self.config['paths']['model_local_path'], self.config['paths']['model_name'], self.config['paths']['tokenizer_name']))
        

        # Push to hub
        fast_tokenizer.push_to_hub(f"{hf_config['username']}/{self.config['paths']['model_name']}")


    def upload_tokenizer_to_hub(self):
        """Upload tokenizer to Hugging Face Hub"""
        try:
            # Login to Hugging Face Hub
            hf_config = self.login_to_huggingface()
            
            # Get tokenizer file path
            tokenizer_file = os.path.join(self.config['paths']['model_local_path'], self.config['paths']['model_name'], self.config['paths']['tokenizer_name'])
            
            # Upload single file to hub
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=tokenizer_file,
                path_in_repo=self.config['paths']['tokenizer_name'],
                repo_id=f"{hf_config['username']}/{self.config['paths']['model_name']}", 
                token=hf_config['token'],
                repo_type="model"
            )
            
            print(f"Successfully uploaded tokenizer.json to {hf_config['username']}/{self.config['paths']['model_name']}")
            
        except Exception as e:
            print(f"Error uploading tokenizer to Hugging Face Hub: {str(e)}")
            raise 


