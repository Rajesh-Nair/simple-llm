# Import required modules
from modules.sequence_generator import generate_sequence
import yaml
import random
from tqdm import tqdm
import os


def load_config(path):
    """
    Load configuration from a YAML file
    
    Args:
        path (str): Path to the YAML config file
        
    Returns:
        dict: Configuration parameters loaded from the file
    """
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def generate_mask(config):
    """
    Generate a random binary mask according to config parameters
    
    Args:
        config (dict): Configuration containing mask parameters
        
    Returns:
        list: Binary mask of specified length with minimum number of ones
    """
    mask = [random.randint(0, 1) for _ in range(config["mask"]["length"])]
    # Recursively regenerate if mask doesn't have minimum required ones
    if sum(mask) < config["mask"]["min_ones"]:
        mask = generate_mask(config)
    return mask

def generate_initial(config):
    """
    Generate initial sequence values according to config parameters
    
    Args:
        config (dict): Configuration containing initial sequence parameters
        
    Returns:
        list: Initial sequence values within specified range
    """
    initial_length = random.randint(config["Initial"]["min_length"], config["Initial"]["max_length"])
    initial = [random.randint(config["Initial"]["min_value"], config["Initial"]["max_value"]) for _ in range(initial_length)]
    return initial

def generate_data(config):
    """
    Generate a complete sequence using mask and initial values
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        list: Generated sequence according to specified parameters
    """
    data = []
    print(f"Generating {config['sequence']['max_rows']} sequences")
    for i in tqdm(range(config["sequence"]["max_rows"])):
        mask = generate_mask(config)
        initial = generate_initial(config)
        sequence = generate_sequence(initial, mask, config["sequence"]["max_length"])
        row = [mask, initial, " ".join(str(x) for x in sequence)]
        data.append(row)
    return data

def save_data(data, config):
    """
    Save generated sequence data to a file
    
    Args:   
        data (list): List of lists containing sequence data
        config (dict): Configuration parameters
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config["sequence"]["path"]), exist_ok=True)
    
    with open(config["sequence"]["path"], "w") as file:
        print(f"Saving data to {config['sequence']['path']}")
        for sequence in tqdm(data):
            # Convert numbers to strings and join with column delimiter
            row = config["sequence"]["column_delimiter"].join(str(x) for x in sequence)
            # Add row delimiter after each sequence
            file.write(row + config["sequence"]["row_delimiter"])
        



if __name__ == "__main__":
    # Load config and generate data when run as main script
    config = load_config("data_config.yaml")
    data = generate_data(config)
    save_data(data, config)




