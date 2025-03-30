# Import required modules
from modules.sequence_generator import *
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
    if config["series"]["mask_length"] <= config["series"]["mask_min_elements"]:
        mask = [1] * config["series"]["mask_length"]
    else:
        mask = [random.randint(0, 1) for _ in range(config["series"]["mask_length"])]
        # Recursively regenerate if mask doesn't have minimum required ones
        if sum(mask) < config["series"]["mask_min_elements"]:   
            mask = generate_mask(config)
    return mask


def generate_data(config):
    """
    Generate a complete sequence using mask and initial values
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        list: Generated sequence according to specified parameters
    """
    data = []
    print(f"Generating {config['series']['max_rows']} sequences")
    total_nums = 0
    
    # Generate all sequences from min_value to max_value
    if config["data_generator"]["sequence_type"] == "sum":
        rows = generate_sum(config["sum"]["min_value"], config["sum"]["max_value"], config["sum"]["retrieve_percent"])        
        for i,row in enumerate(rows):
            data.append([" ".join(str(x) for x in row)])

    # Generate all sequences from min_value to max_value
    elif config["data_generator"]["sequence_type"] == "series":
        for i in tqdm(range(config["series"]["initial_max_value"], config["series"]["initial_min_value"]-1, -1)):
            initial = [i]
            mask = generate_mask(config)
            sequence = generate_series(initial, mask, config["series"]["max_numbers"], 0)
            total_nums += len(sequence)
            row = [" ".join(str(x) for x in sequence)]
            data.append(row)
            for max_len in range(config["series"]["min_numbers"],config["series"]["max_numbers"],1):
                sequence = generate_series(initial, mask,max_len, 0)
                total_nums += len(sequence)
                row = [" ".join(str(x) for x in sequence)]
                data.append(row)
        print(f"Total numbers generated: {total_nums}")

    elif config["data_generator"]["sequence_type"] == "series_max_rows":
        for i in tqdm(range(config["series"]["max_rows"])):
            mask = generate_mask(config)
            initial = [random.randint(config["series"]["min_value"], config["series"]["max_value"])]
            sequence = generate_series(initial, mask, config["series"]["max_numbers"],0)
            row = [" ".join(str(x) for x in sequence)]
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
    os.makedirs(os.path.dirname(config["storage"]["path"]), exist_ok=True)
    
    with open(config["storage"]["path"], "w") as file:
        print(f"Saving data to {config['storage']['path']}")
        for sequence in tqdm(data):
            # Convert numbers to strings and join with column delimiter
            row = config["storage"]["column_delimiter"].join(str(x) for x in sequence)
            # Add row delimiter after each sequence
            file.write(row + config["storage"]["row_delimiter"])
        



if __name__ == "__main__":
    # Load config and generate data when run as main script
    config = load_config("data_config.yaml")
    data = generate_data(config)
    save_data(data, config)




