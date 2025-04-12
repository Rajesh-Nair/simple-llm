# Import required modules
from modules.sequence_generator import *
from modules.data_processor import process
import random
from tqdm import tqdm
import os
from modules.utils import load_config
import re


def generate_mask(data_config):
    """
    Generate a random binary mask according to data_config parameters
    
    Args:
        data_config (dict): data_configuration containing mask parameters
        
    Returns:
        list: Binary mask of specified length with minimum number of ones
    """
    if data_config["series"]["mask_length"] <= data_config["series"]["mask_min_elements"]:
        mask = [1] * data_config["series"]["mask_length"]
    else:
        mask = [random.randint(0, 1) for _ in range(data_config["series"]["mask_length"])]
        # Recursively regenerate if mask doesn't have minimum required ones
        if sum(mask) < data_config["series"]["mask_min_elements"]:   
            mask = generate_mask(data_config)
    return mask


def generate_data(data_config):
    """
    Generate a complete sequence using mask and initial values
    
    Args:
        data_config (dict): data_configuration parameters
        
    Returns:
        list: Generated sequence according to specified parameters
    """
    data = []
    print(f"Generating {data_config['series']['max_rows']} sequences")
    total_nums = 0
    
    # Generate all sequences from min_value to max_value
    if data_config["data_generator"]["sequence_type"] == "sum":
        rows = generate_sum(data_config["sum"]["min_value"], data_config["sum"]["max_value"], data_config["sum"]["retrieve_percent"], data_config["sum"]["max_length"])
        print(f"Generating data for sum")       
        for i,row in tqdm(enumerate(rows)):
            data.append([" ".join(str(x) for x in row)])

    # Generate all sequences from min_value to max_value
    elif data_config["data_generator"]["sequence_type"] == "series":
        for i in tqdm(range(data_config["series"]["initial_max_value"], data_config["series"]["initial_min_value"]-1, -1)):
            initial = [i]
            mask = generate_mask(data_config)
            sequence = generate_series(initial, mask, data_config["series"]["max_numbers"], 0)
            total_nums += len(sequence)
            row = [" ".join(str(x) for x in sequence)]
            data.append(row)
            for max_len in range(data_config["series"]["min_numbers"],data_config["series"]["max_numbers"],1):
                sequence = generate_series(initial, mask,max_len, 0)
                total_nums += len(sequence)
                row = [" ".join(str(x) for x in sequence)]
                data.append(row)
        print(f"Total numbers generated: {total_nums}")

    elif data_config["data_generator"]["sequence_type"] == "series_max_rows":
        for i in tqdm(range(data_config["series"]["max_rows"])):
            mask = generate_mask(data_config)
            initial = [random.randint(data_config["series"]["min_value"], data_config["series"]["max_value"])]
            sequence = generate_series(initial, mask, data_config["series"]["max_numbers"],0)
            row = [" ".join(str(x) for x in sequence)]
            data.append(row)

    return data

def save_data(data, data_config):
    """
    Save generated sequence data to a file
    
    Args:   
        data (list): List of lists containing sequence data
        data_config (dict): data_configuration parameters
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(data_config["storage"]["path"]), exist_ok=True)
    if data_config["storage"]["replace_or_append"] == "replace" :
        write_mode = "w" 
    elif data_config["storage"]["replace_or_append"] == "append" :
        write_mode = "a"
    else :
        write_mode = "w"
    with open(data_config["storage"]["path"], write_mode) as file:
        print(f"Saving data to {data_config['storage']['path']}")
        for sequence in tqdm(data):
            # Convert numbers to strings and join with column delimiter
            row = data_config["storage"]["column_delimiter"].join(str(x) for x in sequence)
            # Add row delimiter after each sequence
            file.write(row + data_config["storage"]["row_delimiter"])
        

def format_data(data, data_config, train_config):
    """
    Fromat the data according to the data_config parameters
    """
    process_object = process(data_config)
    count = 0
    with open(data_config["storage"]["transformed_path"], 'w') as file:
        print(f"Saving data to {data_config['storage']['transformed_path']}")
        for sequence in tqdm(data):
            # Convert numbers to strings and join with column delimiter
            row = process_object.pre_processing(data_config["storage"]["column_delimiter"].join(str(x) for x in sequence)) 
            row = data_config["pre_processing"]["replace_column_delimiter"] + row + data_config["pre_processing"]["replace_column_delimiter"]
            if data_config["data_generator"]["sequence_type"] == "sum":
                try:
                    matches = re.finditer(r'\{}'.format(data_config["pre_processing"]["replace_column_delimiter"]), row)
                    positions = [match.start() for match in matches]
                    input_length = positions[2] + 1
                except:
                    input_length = None
            else:
                    input_length = None
            
            # Add row delimiter after each sequence
            if len(row) <= train_config["model"]["n_positions"]:
                file.write(str(input_length) + ":" + row + data_config["storage"]["row_delimiter"])
                count += 1
    print(f"Total number of sequences saved: {count}")

if __name__ == "__main__":
    # Load data_config and generate data when run as main script
    data_config = load_config("data_config.yaml")
    train_config = load_config("train_config.yaml")
    data = generate_data(data_config)
    save_data(data, data_config)

    # Format the data
    format_data(data, data_config, train_config)




