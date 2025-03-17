from transformers import GPT2LMHeadModel, GPT2Config
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from typing import List, Optional, Tuple
import yaml
import os
from tqdm import tqdm


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
        """Initialize a new GPT2 model with given vocabulary size"""
        model_config = self.config['model']
        config = GPT2Config(
            vocab_size=vocab_size,
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

        return model

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
        
        for epoch in range(train_config['num_epochs']):
            total_loss = 0
            pre_eval_loss = float('inf')
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

    def generate_text(self, model: GPT2LMHeadModel, tokenizer: Tokenizer, prompt: str) -> str:
        """Generate text using the model"""
        gen_config = self.config['generation']
        model = model.to(self.device)
        model.eval()    

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_length=gen_config['max_length'],
                attention_mask=attention_mask
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


class ModelManager:
    def __init__(self, config: dict):
        self.config = config

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


if __name__ == "__main__":
    # Load config
    config = load_config("train_config.yaml")
    
    # Initialize managers
    model_manager = ModelManager(config)
    trainer = GPT2ModelTrainer(config)

    # Load tokenizer
    tokenizer = model_manager.load_tokenizer()

    # Initialize model
    model = trainer.initialize_model(vocab_size=tokenizer.get_vocab_size())

    # Create dataset
    sequences = ["1 1234 42113", "1 1234 42113", "1 1234 42113"]
    dataset = SequenceDataset(sequences, tokenizer, max_length=config['model']['n_positions'])

    # Train model
    model = trainer.train_model(model, dataset, dataset)
    
    # Save checkpoint
    model_manager.save_checkpoint(model, tokenizer)

