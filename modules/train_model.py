from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from typing import List, Optional, Tuple
import yaml
import os
from tqdm import tqdm
from huggingface_hub import login


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
        encoding = self.tokenizer.encode(sequence)
        
        # Pad or truncate to max_length + 1 to have input and target
        input_ids = encoding.ids[:self.max_length+1] 
        if len(input_ids) < self.max_length + 1:
            input_ids.extend([0] * (self.max_length + 1 - len(input_ids)))
            
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

    def initialize_model(self, vocab_size: int) -> GPT2LMHeadModel:

        # Load checkpoint if specified
        if self.config['training']['load_checkpoint'] and self.config['training']['load_checkpoint'].startswith("https://huggingface.co"):
            print(f"Loading model from Hugging Face Hub: {self.config['training']['load_checkpoint']}")

            # Load model and tokenizer using local paths
            model = self.model_manager.load_model_from_hub()
            tokenizer = self.model_manager.load_tokenizer_from_hub()
            
            # Save model and tokenizer to specified paths
            self.model_manager.save_model(model)
            self.model_manager.save_tokenizer(tokenizer)
            self.vocab_size = tokenizer.get_vocab_size()

            return model, tokenizer
        elif self.config['training']['load_checkpoint'] is not None:
            print(f"Loading checkpoint from {self.config['training']['load_checkpoint']}")
            model, tokenizer = self.model_manager.load_checkpoint()
            self.vocab_size = tokenizer.get_vocab_size()
            return model, tokenizer
        else:
            self.vocab_size = vocab_size

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
        # Initialize model manager
        model = GPT2LMHeadModel(config)

        print("Model config: ", config)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

        return model, None

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

        if train_config['upload_to_huggingface']:
            self.model_manager.upload_tokenizer_to_huggingface()
        
        for epoch in range(train_config['num_epochs']):
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
                    self.model_manager.save_checkpoint(model) 
                    if train_config['upload_to_huggingface']:
                        self.model_manager.upload_model_to_huggingface(model)
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

        avg_loss = total_loss / total_samples
        return avg_loss

class TextGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.device = config['training']['device'] if torch.cuda.is_available() else 'cpu'
        self.model_manager = ModelManager(config)
        self.model, self.tokenizer = self.model_manager.load_checkpoint()
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate_text(self, prompt: str, max_length: int = None) -> str:
        """Generate text using the loaded model"""
        if max_length is None:
            max_length = self.config['model']['n_positions']

        input_ids = torch.tensor([self.tokenizer.encode(prompt).ids]).to(self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                attention_mask=attention_mask,
                bos_token_id=None,  # Set to None since config has 'None' as string
                eos_token_id=None,  # Set to None since config has 'None' as string 
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
        return self.tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True).replace(" ##", "")


class ModelManager:
    def __init__(self, config: dict):
        self.config = config
        if self.config['training']['load_checkpoint'] is not None and self.config['training']['load_checkpoint'].startswith("https://huggingface.co"):
            self.hf_config = self.login_to_huggingface()

    def load_tokenizer_from_hub(self) -> Tokenizer:
        """Load tokenizer from Hugging Face Hub"""
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.config['training']['load_checkpoint'].replace("https://huggingface.co/", ""))
        return tokenizer    

    def load_model_from_hub(self) -> GPT2LMHeadModel:
        # Load model and tokenizer from Hugging Face Hub
        model = GPT2LMHeadModel.from_pretrained(self.config['training']['load_checkpoint'].replace("https://huggingface.co/", ""))
        
        return model

    def load_tokenizer(self) -> Tokenizer:
        """Load tokenizer from saved file"""
        return Tokenizer.from_file(os.path.join(self.config['paths']['tokenizer_save_path'], self.config['paths']['tokenizer_file']))

    def save_model(self, model: GPT2LMHeadModel):
        """Save the model to a file"""
        model.save_pretrained(self.config['paths']['model_save_path'])

    def load_model(self) -> GPT2LMHeadModel:
        """Load the model from a file"""
        return GPT2LMHeadModel.from_pretrained(self.config['paths']['model_load_path'])

    def save_tokenizer(self, tokenizer: Tokenizer):
        """Save the tokenizer to a file"""
        os.makedirs(os.path.dirname(self.config['paths']['tokenizer_save_path']), exist_ok=True)
        tokenizer.save(os.path.join(self.config['paths']['tokenizer_save_path'], self.config['paths']['tokenizer_file']))
        
        # Convert tokenizer to Hugging Face format
        _ = self.convert_tokenizer_to_hf_format() 

    def save_checkpoint(self, model: GPT2LMHeadModel, tokenizer: Tokenizer = None):
        """Save the model and tokenizer"""
        self.save_model(model)
        if tokenizer is not None:
            self.save_tokenizer(tokenizer)

    def load_checkpoint(self) -> Tuple[GPT2LMHeadModel, Tokenizer]:
        """Load the model and tokenizer"""
        model = self.load_model()
        tokenizer = self.load_tokenizer()
        return model, tokenizer
    
    def login_to_huggingface(self):
        """Login to Hugging Face Hub"""
        with open("secret.yaml", "r") as f:
            secret_config = yaml.safe_load(f)
            
        hf_config = secret_config["huggingface"]
        login(token=hf_config["token"], add_to_git_credential=True) 
        return hf_config
    
    def convert_tokenizer_to_hf_format(self):
        """Convert tokenizer to Hugging Face format"""
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.config['paths']['tokenizer_save_path']+self.config['paths']['tokenizer_file'])
        fast_tokenizer.save_pretrained(self.config['paths']['tokenizer_save_path'])
        return fast_tokenizer
    
    def upload_tokenizer_to_huggingface(self):

        # Login to Hugging Face Hub
        hf_config = self.login_to_huggingface()

        # Convert tokenizer to Hugging Face format
        fast_tokenizer = self.convert_tokenizer_to_hf_format()  

        # Push to hub
        fast_tokenizer.push_to_hub(f"{hf_config['username']}/{hf_config['model_name']}")


    def upload_model_to_huggingface(self, model: GPT2LMHeadModel):
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


if __name__ == "__main__":
    # Load config
    config = load_config("train_config.yaml")
    
    # Initialize trainer
    trainer = GPT2ModelTrainer(config)

    # Load model manager and tokenizer
    if config['training']['load_checkpoint'] is None:
        model_manager = ModelManager(config)
        tokenizer = model_manager.load_tokenizer()

    # Initialize model
    model, tokenizer = trainer.initialize_model(vocab_size=tokenizer.get_vocab_size())

    # Create dataset
    sequences = ["1 1234 42113", "1 1234 42113", "1 1234 42113"]
    dataset = SequenceDataset(sequences, tokenizer, max_length=config['model']['n_positions'])

    # Train model
    model = trainer.train_model(model, dataset, dataset)
    
    # Save checkpoint
    model_manager.save_checkpoint(model, tokenizer)

