from tokenizers import Tokenizer
from modules.train_model import ModelManager, GPT2ModelTrainer, SequenceDataset
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

# Load config files
train_config = load_config("train_config.yaml")
data_config = load_config("data_config.yaml")   


# Initialize managers
model_manager = ModelManager(train_config)
trainer = GPT2ModelTrainer(train_config)

# Load tokenizer
tokenizer = model_manager.load_tokenizer()

# Initialize model
model = trainer.initialize_model(vocab_size=tokenizer.get_vocab_size())

# Create dataset
sequences = data_preprocessing(data_config["sequence"]["path"])
dataset = SequenceDataset(sequences, tokenizer, max_length=train_config['model']['n_positions'])

# Train test split
train_sequences, test_sequences = dataset.split_train_test(test_size=0.2, random_state=42)

# Train model
model = trainer.train_model(model, train_sequences, test_sequences)

# Generate text
generated_text = trainer.generate_text(model, tokenizer, "1 1 2 3 5", max_length=100)
print(generated_text)









if __name__ == "__main__":
    # Tokenize a sample sequence
    pass


