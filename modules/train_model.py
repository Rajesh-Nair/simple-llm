from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedTokenizerFast
from modules.data_processor import process
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from typing import List, Optional, Tuple
import yaml
import os
from tqdm import tqdm
from huggingface_hub import login, Repository

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from yaml file"""
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class SequenceDataset(Dataset):
    def __init__(self, sequences: List[str], tokenizer: Tokenizer, max_length: int = 128):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # PreTrainedTokenizerFast returns a dictionary with 'input_ids'
        encoding = self.tokenizer(sequence, truncation=True, max_length=self.max_length+1, padding='max_length')
        
        # Get input_ids from the encoding dictionary
        input_ids = encoding['input_ids']
            
        # Split into input and target - target is shifted by 1
        x = torch.tensor(input_ids[:-1])  # Input sequence
        y = torch.tensor(input_ids[1:])   # Target sequence
            
        return x, y
    
    def split_train_test(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
        """Split the dataset into train and test sets"""
        train_size = 1 - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(self, [train_size, test_size], generator=torch.Generator().manual_seed(random_state))
        return train_dataset, test_dataset

class GPT2ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = config['training']['device'] if torch.cuda.is_available() else 'cpu'
        self.model_manager = ModelManager(config)

    def initialize_model(self) -> GPT2LMHeadModel:

        # Load from huggingface checkpoint 
        if self.config['training']['load_checkpoint'] and self.config['training']['load_checkpoint'].startswith("https://huggingface.co"):
            print(f"Loading model from Hugging Face Hub: {self.config['training']['load_checkpoint']}")

            # Load model and tokenizer using local paths
            try :
                self.model_manager.download_repo_from_hub()
                model, tokenizer = self.model_manager.load_checkpoint_from_local()
                self.vocab_size = tokenizer.vocab_size
            except Exception as e:
                print(f"Error downloading repository from Hugging Face Hub: {str(e)}")
                model = self.model_manager.download_model_from_hub()
                tokenizer = self.model_manager.download_fast_tokenizer_from_hub()
            
                # Save model and tokenizer to specified paths
                self.model_manager.save_model_to_local(model)
                self.model_manager.save_fast_tokenizer_to_local(tokenizer)
                self.vocab_size = tokenizer.vocab_size

            return model, tokenizer
        
        # Load from local checkpoint
        elif self.config['training']['load_checkpoint'] is not None:
            print(f"Loading checkpoint from {self.config['training']['load_checkpoint']}")
            model, tokenizer = self.model_manager.load_checkpoint_from_local()
            self.vocab_size = tokenizer.vocab_size
            return model, tokenizer
        
        # Load from local tokenizer and initialize new model
        else:
            tokenizer = self.model_manager.load_fast_tokenizer_from_local()
            self.vocab_size = tokenizer.vocab_size

            """Initialize a new GPT2 model with given vocabulary size"""
            model_config = self.config['model']
            config = GPT2Config(
                vocab_size=self.vocab_size,
                n_positions=model_config['n_positions'],
                n_ctx=model_config['n_positions'],
                n_embd=model_config['n_embd'],
                n_layer=model_config['n_layer'],
                n_head=model_config['n_head'],
                bos_token_id=model_config['bos_token_id'],
                eos_token_id=model_config['eos_token_id'],
                layer_norm_epsilon=model_config['layer_norm_epsilon'],
                activation_function=model_config['activation_function'],
                resid_pdrop=model_config['resid_pdrop'],
                embd_pdrop=model_config['embd_pdrop'],
                attn_pdrop=model_config['attn_pdrop'],
            )
            # Initialize model
            model = GPT2LMHeadModel(config)

            print("Model config: ", config)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params}")
            print(f"Trainable parameters: {trainable_params}")


            return model, tokenizer

    def train_model(
        self,
        model: GPT2LMHeadModel,
        train_dataset: Dataset,
        test_dataset: Dataset
    ) -> GPT2LMHeadModel:
        """Train the GPT2 model on given dataset"""
        train_config = self.config['training']
        
        model = model.to(self.device)
        model.train()
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=train_config['learning_rate']
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=train_config['batch_size'], 
            shuffle=True
        )

        # Upload tokenizer to Hugging Face Hub when training from scratch
        if train_config['upload_to_huggingface'] and self.config['training']['load_checkpoint'] is None:
            self.model_manager.upload_fast_tokenizer_to_hub()
            self.model_manager.upload_tokenizer_to_hub()
        
        for epoch in range(train_config['num_epochs']):
            model.train()
            total_loss = 0
            pre_eval_loss = float('inf')
            early_stopping_counter = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{train_config["num_epochs"]}')
            for input_ids, labels in progress_bar:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
                
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{train_config['num_epochs']}, Average Loss: {avg_loss:.4f}")

            # Evaluate model
            if (epoch + 1) % train_config['eval_interval'] == 0:
                eval_loss = self.evaluate_model(model, test_dataset)
                print(f"Epoch {epoch+1}/{train_config['num_epochs']}, Evaluation Loss: {eval_loss:.4f}")
                if eval_loss < pre_eval_loss:
                    pre_eval_loss = eval_loss
                    self.model_manager.save_model_to_local(model) 
                    if train_config['upload_to_huggingface']:
                        self.model_manager.upload_model_to_hub(model)
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= train_config['early_stopping']:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
        
            # Generate text
            if (epoch + 1) % train_config['generate_text_steps'] == 0:
                text_generator = TextGenerator(self.config)
                generated_text = text_generator.generate_text(train_config['generate_text_input'], max_length=train_config['generate_text_length'])
                print(f"Generated text at epoch {epoch+1}: {generated_text}")
        
        return model    



    def evaluate_model(self, model: GPT2LMHeadModel, dataset: Dataset) -> float:
        """Evaluate the model on the given dataset"""
        model = model.to(self.device)
        model.eval()
        total_loss = 0
        total_samples = 0

        eval_loader = DataLoader(   
            dataset,
            batch_size=self.config['training']['eval_batch_size'],
            shuffle=False
        )

        with torch.no_grad():
            progress_bar = tqdm(eval_loader, desc='Evaluating')
            for input_ids, labels in progress_bar:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                total_samples += input_ids.size(0)
                progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(eval_loader)
        return avg_loss

class TextGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.device = config['training']['device'] if torch.cuda.is_available() else 'cpu'
        self.model_manager = ModelManager(config)
        self.model, self.tokenizer = self.model_manager.load_checkpoint_from_local()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = process(config)

    def generate_text(self, prompt: str, max_length: int = None) -> str:
        """Generate text using the loaded model"""
        if max_length is None:
            max_length = self.config['model']['n_positions']

        # PreTrainedTokenizerFast returns input_ids directly
        input_ids = torch.tensor([self.tokenizer.encode(self.processor.pre_processing(prompt))]).to(self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            if max_length > self.config['model']['n_positions']:
                outputs = input_ids  # Start with the prompt
                i = 0
                while outputs.size(1) < max_length and i < 150: #Max 150 tokens
                    # Get the last n_positions-1 tokens as context
                    context = outputs[:, -(self.config['model']['n_positions']-1):]
                    # Update attention mask for the context
                    context_mask = torch.ones_like(context)
                    
                    # Generate next token
                    output = self.model.generate(
                        context,
                        max_length=context.size(1) + 1,  # Only generate one new token
                        attention_mask=context_mask,
                        bos_token_id=None,
                        eos_token_id=None,
                        do_sample=True,
                        top_k=1,
                        top_p=0.95,
                        temperature=0.01
                    )
                    
                    # Append only the newly generated token
                    new_token = output[:, -1:]
                    outputs = torch.cat([outputs, new_token], dim=1)
                    i += 1
            else :
                outputs = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    attention_mask=attention_mask,
                    bos_token_id=None,  # Set to None since config has 'None' as string
                    eos_token_id=None,  # Set to None since config has 'None' as string 
                    do_sample=True,
                    top_k=1,
                    top_p=0.95,
                    temperature=0.01
                )
        return self.processor.post_processing(self.tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True).replace(" ", ""))


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


    def load_model_from_local(self) -> GPT2LMHeadModel:
        """Load the model from a file"""
        return GPT2LMHeadModel.from_pretrained(self.config['paths']['model_load_path'])
    
    
    def load_checkpoint_from_local(self) -> Tuple[GPT2LMHeadModel, PreTrainedTokenizerFast]:
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
        


    def save_model_to_local(self, model: GPT2LMHeadModel):
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
    

    def download_model_from_hub(self) -> GPT2LMHeadModel:
        # Load model and tokenizer from Hugging Face Hub
        model = GPT2LMHeadModel.from_pretrained(self.config['training']['load_checkpoint'].replace("https://huggingface.co/", ""))
        
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
    
    def upload_model_to_hub(self, model: GPT2LMHeadModel):
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


if __name__ == "__main__":
    # Load config
    config = load_config("train_config.yaml")
    
    # Initialize trainer
    trainer = GPT2ModelTrainer(config)

    # Load model manager and tokenizer
    if config['training']['load_checkpoint'] is None:
        model_manager = ModelManager(config)

    # Initialize model
    model, tokenizer = trainer.initialize_model()

    # Create dataset
    sequences = ["1 1234 42113", "1 1234 42113", "1 1234 42113"]
    dataset = SequenceDataset(sequences, tokenizer, max_length=config['model']['n_positions'])

    # Train model
    model = trainer.train_model(model, dataset, dataset)
    
    # Save checkpoint
    model_manager.save_model_to_local(model)

