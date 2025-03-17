from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os


def train_bpe_tokenizer(file_path: str, vocab_size: int = 1000, output_dir: str = "../tokenizer") -> None:
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
    
    # Initialize trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|UNK|>"],
        min_frequency=2,
        continuing_subword_prefix="##"
    )
    
    # Train tokenizer
    tokenizer.train(files=[file_path], trainer=trainer)
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the tokenizer files
    tokenizer.save(f"{output_dir}/tokenizer.json")
    


if __name__ == "__main__":
    train_bpe_tokenizer(file_path="../synthetic_data/sequences.txt", vocab_size=1000, output_dir="../synthetic_data/tokenizer")
