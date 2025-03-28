from transformers import GPT2LMHeadModel, GPT2Config
from modules.data_processor import process
from modules.model_mgr import ModelManager
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from typing import List, Optional, Tuple
import yaml
import os
from tqdm import tqdm
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup
import wandb

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
        encoding = self.tokenizer(sequence, truncation=True, max_length=self.max_length+1, padding='max_length', padding_side='left')
        
        # Get input_ids and create attention mask
        input_ids = encoding['input_ids']
        attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in input_ids]
            
        # Split into input and target - target is shifted by 1
        x = torch.tensor(input_ids[:-1])  # Input sequence
        y = torch.tensor(input_ids[1:])   # Target sequence
        mask = torch.tensor(attention_mask[:-1])  # Attention mask for input
            
        return x, y, mask
    
    def split_train_test(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
        """Split the dataset into train and test sets"""
        train_size = 1 - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(self, [train_size, test_size], generator=torch.Generator().manual_seed(random_state))
        return train_dataset, test_dataset

class GPT2ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.model_manager = ModelManager(config)        
        self.processor = process(config)
        
        # Initialize wandb if enabled in config
        if self.config.get('wandb', {}).get('enabled', False):
            with open("secret.yaml", "r") as f:
                secret_config = yaml.safe_load(f)
            wandb_config = secret_config.get("wandb", {})
            if self.accelerator.is_main_process:
                wandb.login(key=wandb_config.get("api_key"))
                wandb.init(
                    project=wandb_config.get("project_name"),
                    entity=wandb_config.get("entity"),
                    config=self.config
                )
        

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

            # Store tokenizer
            self.tokenizer = tokenizer
            return model, tokenizer
        
        # Load from local checkpoint
        elif self.config['training']['load_checkpoint'] is not None:
            print(f"Loading checkpoint from {self.config['training']['load_checkpoint']}")
            model, tokenizer = self.model_manager.load_checkpoint_from_local()
            self.vocab_size = tokenizer.vocab_size

            # Store tokenizer
            self.tokenizer = tokenizer
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

            # Store tokenizer
            self.tokenizer = tokenizer
            return model, tokenizer

    def train_model(
        self,
        model: GPT2LMHeadModel,
        train_dataset: Dataset,
        test_dataset: Dataset
    ) -> GPT2LMHeadModel:
        """Train the GPT2 model on given dataset"""
        train_config = self.config['training']
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
        
        # Create sampler for distributed training
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=self.accelerator.num_processes,
            rank=self.accelerator.process_index,
            shuffle=True
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=train_config['per_device_batch_size'],
            sampler=train_sampler,
            num_workers=train_config['num_workers'],
            pin_memory=True
        )

        # Calculate total training steps
        total_steps = len(train_loader) * train_config['num_epochs']
        
        # Create learning rate scheduler with warmup
        # total_steps = num_epochs * steps_per_epoch, where steps_per_epoch = len(train_loader)
        # This means each step is one batch of training data
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=train_config['warmup_steps'], 
            num_training_steps=total_steps  # total_steps = num_epochs * len(train_loader)
        )

        # Prepare for distributed training
        model, optimizer, train_loader, scheduler = self.accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )

        # Upload tokenizer to Hugging Face Hub when training from scratch
        if train_config['upload_to_huggingface'] and self.config['training']['load_checkpoint'] is None:
            self.model_manager.upload_fast_tokenizer_to_hub()
            self.model_manager.upload_tokenizer_to_hub()
        try:
            pre_eval_loss = self.evaluate_model(model, test_dataset)
        except Exception as e:
            print(f"Error evaluating model: {str(e)}")
            pre_eval_loss = float('inf')
        early_stopping_counter = 0
        for epoch in range(train_config['num_epochs']):
            model.train()
            train_sampler.set_epoch(epoch)  # Important for proper shuffling
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{train_config["num_epochs"]}')
            optimizer.zero_grad()  # Zero gradients at start of epoch
            
            for step, (input_ids, labels, attention_mask) in enumerate(progress_bar):
                outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss / train_config['gradient_accumulation_steps']
                
                self.accelerator.backward(loss)
                
                # Only update weights after accumulating enough gradients
                if (step + 1) % train_config['gradient_accumulation_steps'] == 0:
                    # Clip gradients
                    self.accelerator.clip_grad_norm_(model.parameters(), train_config['max_grad_norm'])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * train_config['gradient_accumulation_steps']
                progress_bar.set_postfix({'loss': loss.item() * train_config['gradient_accumulation_steps']})
                
                # Log metrics to wandb if enabled
                if self.config.get('wandb', {}).get('enabled', False) and self.accelerator.is_main_process:
                    wandb.log({
                        "train_loss": loss.item() * train_config['gradient_accumulation_steps'],
                        "learning_rate": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "global_step": epoch * len(train_loader) + step
                    })
                
            # Gather loss from all processes
            total_loss = self.accelerator.gather(torch.tensor(total_loss).to(self.device)).mean().item()
            avg_loss = total_loss / len(train_loader)
            
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{train_config['num_epochs']}, Average Loss: {avg_loss:.4f}")

            # Evaluate model
            if (epoch + 1) % train_config['eval_interval'] == 0:
                eval_loss = self.evaluate_model(model, test_dataset)
                if self.accelerator.is_main_process:
                    print(f"Epoch {epoch+1}/{train_config['num_epochs']}, Evaluation Loss: {eval_loss:.4f}")
                    
                    # Log evaluation metrics to wandb if enabled
                    if self.config.get('wandb', {}).get('enabled', False):
                        wandb.log({
                            "eval_loss": eval_loss,
                            "epoch": epoch
                        })
                    
                    if eval_loss < pre_eval_loss:
                        pre_eval_loss = eval_loss
                        # Unwrap model before saving
                        unwrapped_model = self.accelerator.unwrap_model(model)
                        self.model_manager.save_model_to_local(unwrapped_model) 
                        if train_config['upload_to_huggingface']:
                            self.model_manager.upload_model_to_hub(unwrapped_model)
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                        if early_stopping_counter >= train_config['early_stopping']:
                            print(f"Early stopping at epoch {epoch+1}")
                            break
        
            # Generate text only on main process
            if self.accelerator.is_main_process and train_config['generate_text_steps'] and (epoch + 1) % train_config['generate_text_steps'] == 0:
                text_generator = TextGenerator(self.config)
                generated_text = text_generator.generate_text(train_config['generate_text_input'], max_length=train_config['generate_text_length'])
                print(f"Generated text at epoch {epoch+1}: {generated_text}")
                
                # Log generated text to wandb if enabled
                if self.config.get('wandb', {}).get('enabled', False):
                    wandb.log({
                        "generated_text": generated_text,
                        "epoch": epoch
                    })
            
            # Wait for all processes to sync up
            self.accelerator.wait_for_everyone()
        
        # Finish wandb run if enabled
        if self.config.get('wandb', {}).get('enabled', False) and self.accelerator.is_main_process:
            wandb.finish()
            
            

    def evaluate_model(self, model: GPT2LMHeadModel, dataset: Dataset) -> float:
        """Evaluate the model on the given dataset"""
        model.eval()
        total_loss = 0
        total_samples = 0

        # Create sampler for distributed evaluation
        eval_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=self.accelerator.num_processes,
            rank=self.accelerator.process_index,
            shuffle=False
        )

        eval_loader = DataLoader(   
            dataset,
            batch_size=self.config['training']['per_device_eval_batch_size'],
            sampler=eval_sampler,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )

        # Prepare for distributed evaluation
        eval_loader = self.accelerator.prepare(eval_loader)

        with torch.no_grad():
            progress_bar = tqdm(eval_loader, desc='Evaluating')
            for input_ids, labels, attention_mask in progress_bar:
                outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss
                total_loss += loss.item()
                total_samples += input_ids.size(0)
                progress_bar.set_postfix({'loss': loss.item()})

        # Gather and average loss across all processes
        total_loss = self.accelerator.gather(torch.tensor(total_loss).to(self.device)).mean().item()
        avg_loss = total_loss / len(eval_loader)
        return avg_loss
 
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





if __name__ == "__main__":
    # Initialize accelerator
    accelerator = Accelerator()

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
    trainer.train_model(model, dataset, dataset)
