from tokenizers import Tokenizer
from modules.train_model import ModelManager, GPT2ModelTrainer, SequenceDataset, TextGenerator
import yaml

# function to load config files
def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from yaml file"""
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# function to preprocess data
def data_preprocessing(file_path: str) :
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 3:
                sequences.append(parts[2].strip())
    return sequences

def train(train_config_path: str = "train_config.yaml", data_config_path: str = "data_config.yaml"):
    """Train the model using the specified config files"""
    # Load config files
    train_config = load_config(train_config_path)
    data_config = load_config(data_config_path)   

    # Initialize trainer
    trainer = GPT2ModelTrainer(train_config)

    # Load model manager and tokenizer
    if train_config['training']['load_checkpoint'] is None:
        # Load tokenizer
        model_manager = ModelManager(train_config)
        tokenizer = model_manager.load_tokenizer()

        # Initialize model
        model, _ = trainer.initialize_model(vocab_size=tokenizer.get_vocab_size())
    else :
        # Load model and tokenizer from checkpoint
        model, tokenizer = trainer.initialize_model(vocab_size=None)

    # Create dataset
    sequences = data_preprocessing(data_config["sequence"]["path"])
    dataset = SequenceDataset(sequences, tokenizer, max_length=train_config['model']['n_positions'])

    # Train test split
    train_sequences, test_sequences = dataset.split_train_test(test_size=0.2, random_state=42)

    # Train model
    model = trainer.train_model(model, train_sequences, test_sequences)
    
    return model, tokenizer


if __name__ == "__main__":
    # Train the model
    model, tokenizer = train()

    # Generate text
    # Load config files
    train_config = load_config("train_config.yaml")
    text_generator = TextGenerator(train_config)
    generated_text = text_generator.generate_text("1", max_length=100)
    print(generated_text)



