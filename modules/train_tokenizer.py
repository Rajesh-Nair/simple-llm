from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
from transformers import PreTrainedTokenizerFast
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

def convert_tokenizer_to_hf_format(tokenizer):
    """Convert tokenizer to Hugging Face format"""
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    return fast_tokenizer


def train_bpe_tokenizer(file_path: str, tokenizer_config: dict, output_dir: str = "../tokenizer") -> None:
    """
    Train a BPE tokenizer on input text file and save vocabulary and merge files
    
    Args:
        file_path (str): Path to input text file
        vocab_size (int): Maximum vocabulary size
        output_dir (str): Directory to save tokenizer files
    """
        
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<|UNK|>", continuing_subword_prefix="##"))
    
    # Configure pre-tokenization
    tokenizer.pre_tokenizer = Whitespace()

    # Add pad token
    tokenizer.pad_token = "<|pad|>"
    tokenizer.eos_token = "<|pad|>"
    
    # Initialize trainer
    trainer = BpeTrainer(
        **tokenizer_config
    )
    

    # Train tokenizer
    tokenizer.train(files=[file_path], trainer=trainer)

    fast_tokenizer = convert_tokenizer_to_hf_format(tokenizer)
    

    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the tokenizer files
    tokenizer.save(f"{output_dir}/tokenizer.json")

    fast_tokenizer.save_pretrained(output_dir)
    

class SequenceDataset(Dataset):
    def __init__(self, sequences: List[str], tokenizer: Tokenizer, max_length: int = 128, split_input_length = False):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split_input_length = split_input_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_length, sequence = self.sequences[idx].split(":")
        if self.split_input_length and not input_length:
            raise ValueError("Input length is not provided")
        # PreTrainedTokenizerFast returns a dictionary with 'input_ids'
        encoding = self.tokenizer(sequence, truncation=True, max_length=self.max_length+1, padding='max_length', padding_side='right')
        
        # Get input_ids and create attention mask
        input_ids = encoding['input_ids']
        attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in input_ids]
            
        # Split into input and target - target is shifted by 1
        x = torch.tensor(input_ids[:-1])  # Input sequence
        y = torch.tensor(input_ids[1:])   # Target sequence
        mask = torch.tensor(attention_mask[:-1])  # Attention mask for input

        # Convert target y to ignore pad positions with -100
        y[y == self.tokenizer.pad_token_id] = -100

        # Assuming no merge of input bits during tokenization
        if input_length and self.split_input_length:
            print("input_length : ", input_length)
            input_length = int(input_length)
            y[:input_length-1] = -100
            
        return x, y, mask
    
    
    def split_train_test(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
        """Split the dataset into train and test sets"""
        train_size = 1 - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(self, [train_size, test_size], generator=torch.Generator().manual_seed(random_state))
        return train_dataset, test_dataset

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    train_bpe_tokenizer(file_path="../synthetic_data/sequences.txt", vocab_size=1000, output_dir="../synthetic_data/tokenizer")
