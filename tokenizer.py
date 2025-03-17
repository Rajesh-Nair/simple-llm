from modules.train_tokenizer import train_bpe_tokenizer
import os
import yaml

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def data_preprocessing(function: callable, *args, **kwargs) :

    def extract_sequences(file_path: str, *args, **kwargs) :
        sequences = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    sequences.append(parts[2].strip())

            # Write sequences to temporary file for tokenizer training
        temp_file = 'temp_sequences.txt'
        with open(temp_file, 'w') as f:
            for seq in sequences:
                f.write(f"{seq}\n")
        
        print(f"Temporary file created: {temp_file}")

        # Call the function with the temporary file 
        output = function(file_path=temp_file, *args, **kwargs)

            
        # Clean up temporary file
        print(f"Removing temporary file: {temp_file}")
        os.remove(temp_file)

        return output
        
    return extract_sequences

@data_preprocessing
def build_tokenizer(file_path: str, vocab_size: int = 1000, output_dir: str = "../tokenizer") -> None:
    """
    Train a BPE tokenizer on input text file and save vocabulary and merge files
    
    Args:
        file_path (str): Path to input text file
        vocab_size (int): Maximum vocabulary size
        output_dir (str): Directory to save tokenizer files
    """
    print(f"Building tokenizer with file: {file_path} and output directory: {output_dir}")
    train_bpe_tokenizer(file_path=file_path, vocab_size=vocab_size, output_dir=output_dir)


if __name__ == "__main__":
    data_config = load_config("data_config.yaml")
    train_config = load_config("train_config.yaml")
    
    build_tokenizer(
        file_path=data_config["sequence"]["path"], # Path to the input text file
        output_dir=train_config["paths"]["tokenizer_save_path"], # Path to save the tokenizer
        vocab_size=train_config.get("tokenizer", {}).get("vocab_size", 1000)  # Default 1000 if not specified
    )

