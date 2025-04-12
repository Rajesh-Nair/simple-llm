from modules.model_mgr import ModelManager
import torch
from modules.utils import load_config
from modules.train_tokenizer import SequenceDataset

# Load config
config = load_config("train_config.yaml")


# Initialize model manager and load model/tokenizer
def load_model_and_tokenizer(config):
    model_manager = ModelManager(config)
    #model = model_manager.load_model_from_local()
    ##tokenizer = model_manager.load_fast_tokenizer_from_local()
    model = model_manager.download_model_from_hub()
    tokenizer = model_manager.download_fast_tokenizer_from_hub()
    print("loaded model and tokenizer")
    return model, tokenizer

def data_reader(file_path: str, train_config_path: str = "train_config.yaml") :
    sequences = []
    train_config = load_config(train_config_path)
    with open(file_path, 'r') as f:        
        for i, line in enumerate(f):
            sequence = line.strip()
            if len(sequence.split(":")[1]) <= train_config['model']['n_positions']:
                sequences.append(sequence)

    return sequences

def load_test_dataset(config, tokenizer):
    test_data = data_reader(config["data_generator"]["test_data_path"])
    test_dataset = SequenceDataset(test_data, tokenizer, max_length=config["model"]["n_positions"])
    return test_dataset

def evaluate_model(model, tokenizer, test_dataset):
    model.eval()
    with torch.no_grad():
        for i in range(len(test_dataset)):
            input_ids, label_ids, attention_mask = test_dataset[i]
            input_ids = input_ids.unsqueeze(0)
            label_ids = label_ids.unsqueeze(0)  
            attention_mask = attention_mask.unsqueeze(0)
            position_ids = model._get_block_positions(input_ids)
            output = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=label_ids)
            loss = output.loss
            print(f"Loss: {loss.item():.6f}")

if __name__ == "__main__":
    # Load test dataset
    test_dataset = SequenceDataset(test_data, tokenizer, max_length=config["model"]["n_positions"])
    evaluate_model(model, tokenizer, test_dataset)

