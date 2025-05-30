#! /bin/bash

# Download from Git
git clone -b simple-add1 https://github.com/Rajesh-Nair/simple-llm.git

# Go to the project
cd simple-llm

# Create secret.yaml file in the project
cp secret_templaye.yaml secret.yaml

# Change the token in the secret.yaml file under github and huggingface sections


# Install dependencies
pip install -r requirements.txt

# cache the credentials
git config --global credential.helper store
huggingface-cli login

# set tokenizer parallelism
export TOKENIZERS_PARALLELISM=true

# Data Generation
python3 data_generator.py

# Train tokenizer
python3 tokenizer.py

# Training
accelerate launch --config_file "accelerate_config.yaml" trainer.py

