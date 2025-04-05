from tokenizers import Tokenizer
from modules.train_model import ModelManager, GPT2ModelTrainer, SequenceDataset, TextGenerator
from modules.data_processor import process
import yaml
import re

# function to load config files
def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from yaml file"""
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def sub_sequences(sequence: str, delimiter: str) :
    delim_pos = [m.start() for m in re.finditer(re.escape(delimiter), sequence)]
    seq = []
    for i in range(delim_pos[2]+1, len(sequence)-1):
        seq.append(sequence[:i+1])
    return seq

# function to preprocess data
def data_preprocessing(file_path: str, processor: process, train_config_path: str = "train_config.yaml") :
    sequences = []
    train_config = load_config(train_config_path)
    with open(file_path, 'r') as f:        
        for i, line in enumerate(f):
            sequence = line.strip()
            if len(sequence) <= train_config['model']['n_positions']:
                sequences.append(sequence)

    return sequences

def train(train_config_path: str = "train_config.yaml", data_config_path: str = "data_config.yaml"):
    """Train the model using the specified config files"""
    # Load config files
    train_config = load_config(train_config_path)
    data_config = load_config(data_config_path)   

    # Initialize trainer
    trainer = GPT2ModelTrainer(train_config)

    # Initialize model and tokenizer
    model, tokenizer = trainer.initialize_model()

    # Load dataset    
    processor = process(train_config)
    sequences = data_preprocessing(data_config["storage"]["transformed_path"], processor)
    print(f"Number of sequences: {len(sequences)}")
    
    # Shuffle sequences before creating dataset
    import random
    random.seed(42)
    random.shuffle(sequences)
    
    # Create dataset
    dataset = SequenceDataset(sequences, tokenizer, max_length=train_config['model']['n_positions'])

    # Train test split
    train_sequences, test_sequences = dataset.split_train_test(test_size=0.3, random_state=42)

    # Train the model
    trainer.train_model(model, train_sequences, test_sequences)
    
    


if __name__ == "__main__":
    # Train the model
    train()

    # Generate text
    # Load config files
    # train_config = load_config("train_config.yaml")
    # text_generator = TextGenerator(train_config)
    # generated_text = text_generator.generate_text("1234 1234 ", max_length=100)
    # print(generated_text)



