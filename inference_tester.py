from modules.model_mgr import ModelManager
from modules.train_model import load_config
from modules.model_inference import TextGenerator

# Load config
config = load_config("train_config.yaml")



# Initialize model manager and load model/tokenizer
model_manager = ModelManager(config)
model = model_manager.download_model_from_hub()
tokenizer = model_manager.download_fast_tokenizer_from_hub()
print("saved model and tokenizer")
model_manager.save_model_to_local(model)
model_manager.save_fast_tokenizer_to_local(tokenizer)


# Initialize text generator for inference
text_generator = TextGenerator(config)

prompt = "+1+01+"
text = text_generator.generate_text(prompt, max_length=5)
print("Generated text : ", text)