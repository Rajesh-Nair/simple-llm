# Import required modules
from modules.sequence_generator import *
from modules.data_processor import process
import random
from tqdm import tqdm
import os
from modules.utils import load_config
import re



def generate_data(data_config):
    """
    Generate a complete sequence using mask and initial values
    
    Args:
        data_config (dict): data_configuration parameters
        
    Returns:
        list: Generated sequence according to specified parameters
    """
    data = []
    total_nums = 0
    
    # Generate all sequences from min_value to max_value
    if data_config["data_generator"]["sequence_type"] == "sum":
        rows = generate_sum(data_config["sum"]["min_value"], data_config["sum"]["max_value"], data_config["sum"]["retrieve_percent"], data_config["sum"]["max_length"])
        print(f"Generating data for sum")       
        for i,row in tqdm(enumerate(rows)):
            data.append([" ".join(str(x) for x in row)])

    

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
    process_object = process(train_config)
    count = 0
    with open(data_config["storage"]["transformed_path"], 'w') as file:
        print(f"Saving data to {data_config['storage']['transformed_path']}")
        for sequence in tqdm(data):
            # Convert numbers to strings and join with column delimiter
            row = process_object.pre_processing(data_config["storage"]["column_delimiter"].join(str(x) for x in sequence)) 
            row = train_config["pre_processing"]["replace_column_delimiter"] + row + train_config["pre_processing"]["replace_column_delimiter"]
            if data_config["data_generator"]["sequence_type"] == "sum" and train_config["pre_processing"]["split_output"]:
                try:
                    matches = re.finditer(r'\{}'.format(train_config["pre_processing"]["replace_column_delimiter"]), row)
                    positions = [match.start() for match in matches]
                    num1_length = positions[1] + 1
                    input_length = positions[2] + 1
                    num2_length = input_length - num1_length
                    output_length = len(row) - input_length
                except:
                    num1_length = None
                    input_length = None
                    num2_length = None
                    output_length = None
            else:
                num1_length = None
                input_length = None
                num2_length = None
                output_length = None
            
            if train_config["pre_processing"]["shift_method"] == "standard":
                shift_label = 1
            else:
                shift_label = min(num2_length, output_length)

            # Add row delimiter after each sequence
            if len(row) - shift_label <= train_config["model"]["n_positions"]:
                if data_config["data_generator"]["sequence_type"] == "sum" and train_config["pre_processing"]["split_output"]:
                    if input_length and num1_length:
                        file.write(row[:num1_length] + ":" + row[num1_length:input_length] + ":" + row[input_length:] + data_config["storage"]["row_delimiter"])
                        count += 1
                else:
                    file.write(row + data_config["storage"]["row_delimiter"])
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




