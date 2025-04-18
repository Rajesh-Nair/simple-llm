from modules.train_tokenizer import train_bpe_tokenizer
from modules.data_processor import process
import os
from modules.utils import load_config

def data_preprocessing(function: callable, *args, **kwargs) :

    def extract_sequences(file_path: str, *args, **kwargs) :
        sequences = []


        with open(file_path, 'r') as f:
            for line in f:
                sequences.append(line.strip())

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
def build_tokenizer(file_path: str, tokenizer_config: dict, output_dir: str = "../tokenizer") -> None:
    """
    Train a BPE tokenizer on input text file and save vocabulary and merge files
    
    Args:
        file_path (str): Path to input text file
        vocab_size (int): Maximum vocabulary size
        output_dir (str): Directory to save tokenizer files
    """
    print(f"Building tokenizer with file: {file_path} and output directory: {output_dir}")
    train_bpe_tokenizer(file_path=file_path, tokenizer_config=tokenizer_config, output_dir=output_dir)


if __name__ == "__main__":
    data_config = load_config("data_config.yaml")
    train_config = load_config("train_config.yaml")
    
    if train_config['training']['load_checkpoint'] is not None:
        print("Loading checkpoint - skipping tokenizer training")
    else:
        build_tokenizer(
            file_path=data_config["storage"]["transformed_path"], # Path to the input text file
            output_dir=os.path.join(train_config["paths"]["model_local_path"], train_config["paths"]["model_name"]), # Path to save the tokenizer
            tokenizer_config=train_config["tokenizer"]
        )

