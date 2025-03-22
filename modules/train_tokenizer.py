from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
from transformers import PreTrainedTokenizerFast

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
    


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    train_bpe_tokenizer(file_path="../synthetic_data/sequences.txt", vocab_size=1000, output_dir="../synthetic_data/tokenizer")
