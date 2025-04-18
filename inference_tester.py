from modules.utils import load_config
from modules.model_mgr import ModelManager
from modules.model_inference import TextGenerator
from modules.data_processor import process
import random
from tqdm import tqdm

# Load config
train_config = load_config("train_config.yaml")
data_config = load_config("data_config.yaml")

# Initialize model manager and load model/tokenizer
model_manager = ModelManager(train_config)
#model = model_manager.load_model_from_local()#
#tokenizer = model_manager.load_fast_tokenizer_from_local()
model = model_manager.download_model_from_hub()
tokenizer = model_manager.download_fast_tokenizer_from_hub()
print("saved model and tokenizer")
model_manager.save_model_to_local(model)
model_manager.save_fast_tokenizer_to_local(tokenizer)


text_generator = TextGenerator(train_config)

total_correct = 0
total_pairs = 10000

tokens_rate = 1
if train_config['pre_processing']['shift_method'] == "full":
    tokens_rate = None
elif train_config['pre_processing']['shift_method'] == "standard":
    tokens_rate = 1
else:
    try:
        tokens_rate = int(train_config['pre_processing']['shift_method']) + 1
    except:
        raise ValueError(f"Invalid shift method: {train_config['pre_processing']['shift_method']}")

for _ in tqdm(range(total_pairs)):
    num1 = random.randint(0, 9999)
    num2 = random.randint(0, 9999)

    prompt = f"{num1} {num2}"
    processor = process(train_config)
    delimiter = train_config["pre_processing"]["replace_column_delimiter"]
    prompt = delimiter + processor.pre_processing(prompt) + delimiter

    if train_config['pre_processing']['shift_method'] == "full":
        tokens_rate = len(processor.pre_processing(f"{num2}") + delimiter)
    #print("Prompt : ", prompt)

    # Initialize text generator for inference
    text = text_generator.generate_tokens(prompt, max_length=16, tokens_rate=tokens_rate)
    #print("Generated text : ", text)


    #print("Text: ", text)
    output = processor.post_processing(text.strip())
    #print(f"Sum of {num1} and {num2} is : {output}")
    if int(output) == num1 + num2:
        total_correct += 1

print("tokens_rate: ", tokens_rate)
print(f"Total correct : {total_correct} out of {total_pairs}")
print(f"Accuracy : {total_correct/total_pairs*100}%")
