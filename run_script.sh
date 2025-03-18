#! /bin/bash

# Git Credentials (Replace with your own credentials)
export GITHUB_TOKEN=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
git config --global user.email xxxxxxx@gmail.com
git config --global user.name xxxxxxx


# Download from Git
git clone -b run_pod https://$GITHUB_TOKEN@github.com/Rajesh-Nair/simple-llm.git

# Go to the project
cd simple-llm

# Create secret.yaml file in the project
cp secret_templaye.yaml secret.yaml

# Change the token in the secret.yaml file under github and huggingface sections
export GITHUB_TOKEN=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
export HUGGINGFACE_TOKEN=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
sed -i 's/GITHUB_TOKEN/'$GITHUB_TOKEN'/' secret.yaml
sed -i 's/HUGGINGFACE_TOKEN/'$HUGGINGFACE_TOKEN'/' secret.yaml


# Install dependencies
pip install -r requirements.txt

# Data Generation
python3 data_generator.py

# Train tokenizer
python3 tokenizer.py

# Training
python3 trainer.py

# Upload to git from credentials in secret.yaml
git config --global user.name $(cat secret.yaml | grep -oP 'username: \K[^ ]+')
git config --global user.email $(cat secret.yaml | grep -oP 'email: \K[^ ]+')
git config --global user.token $(cat secret.yaml | grep -oP 'token: \K[^ ]+')

# Add all files to git
git add .
git commit -m "Update model"
git push


