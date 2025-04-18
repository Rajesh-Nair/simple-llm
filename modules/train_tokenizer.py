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
    def __init__(self, sequences: List[str], tokenizer: Tokenizer, max_length: int = 128, shift_method: str = "standard"):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shift_method = shift_method

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x_seq, y_seq, z_seq = self.sequences[idx].split(":")
        input_length = len("".join([x_seq, y_seq]))

        if self.shift_method == "standard":
            self.shift_label = 0
        elif self.shift_method == "full":
            self.shift_label = len(y_seq)
        else:
            try:
                self.shift_label = int(self.shift_method)
            except:
                raise ValueError(f"Invalid shift method: {self.shift_method}")

        sequence = "".join([x_seq, y_seq, z_seq])

        # PreTrainedTokenizerFast returns a dictionary with 'input_ids'
        sequence_encoded = self.tokenizer(sequence, truncation=True, max_length=self.max_length+self.shift_label, padding='max_length', padding_side='right')
        
        # Get input_ids and create attention mask
        sequence_ids = sequence_encoded['input_ids']
        sequence_attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in sequence_ids]
            
        # Split into input and target - target is shifted by shift_label
        if self.shift_label == 0:
            x = torch.tensor(sequence_ids)  # Input sequence
            x_mask = torch.tensor(sequence_attention_mask)  # Attention mask for input
        else:
            x = torch.tensor(sequence_ids[:-1*self.shift_label])  # Input sequence
            x_mask = torch.tensor(sequence_attention_mask[:-1*self.shift_label])  # Attention mask for input

        # Target sequence
        y = torch.tensor(sequence_ids[self.shift_label:])

        # Convert target y to ignore pad positions with -100
        y[y == self.tokenizer.pad_token_id] = -100

        # Do not train on the input sequence
        y[:input_length-self.shift_label] = -100
            
        return x, y, x_mask
    
    
    def split_train_test(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
        """Split the dataset into train and test sets"""
        train_size = 1 - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(self, [train_size, test_size], generator=torch.Generator().manual_seed(random_state))
        return train_dataset, test_dataset

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    train_bpe_tokenizer(file_path="../synthetic_data/sequences.txt", vocab_size=1000, output_dir="../synthetic_data/tokenizer")
