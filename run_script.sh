#! /bin/bash

# Download from Git
git clone -b COT https://$GITHUB_TOKEN@github.com/Rajesh-Nair/simple-llm.git

# Go to the project
cd simple-llm

# Create secret.yaml file in the project
cp secret_templaye.yaml secret.yaml

# Change the token in the secret.yaml file under github and huggingface sections

# cache the credentials
git config --global credential.helper store
huggingface-cli login

# Install dependencies
pip install -r requirements.txt

# Data Generation
python3 data_generator.py

# Train tokenizer
python3 tokenizer.py

# Training
python3 trainer.py


# Add all files to git
git add .
git commit -m "Update model"
git push


