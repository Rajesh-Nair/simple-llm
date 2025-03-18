# Simple LLM Training with GPT-2 Architecture

This repository demonstrates how to train a Language Learning Model (LLM) from scratch using the GPT-2 architecture. The model is trained on numerical sequences to learn and predict patterns.

## Overview
The project implements a complete ML pipeline including:
- Synthetic dataset generation for number sequences
- Custom tokenizer training 
- Model training using GPT-2 architecture
- Inference capabilities

## Dataset Generator
The data generator creates synthetic number sequences based on patterns defined in `data_config.yaml`. Key configuration parameters include:

1. Mask settings:
   - Binary mask sequence length: 4
   - Minimum required ones: 2

2. Initial sequence parameters:
   - Length range: 1-1
   - Value range: 0-20

3. Output sequence settings:
   - Maximum length: 100
   - Number of sequences: 100,000
   - Output path: ../simple-llm-data/sequences.txt
   - Delimiters: | (columns), \n (rows)

To generate the dataset:
1. Configure the data generation parameters in `data_config.yaml`
2. Run `python3 data_generator.py`

## Training
The model training process consists of:

1. Training the tokenizer:
   ```bash
   python3 tokenizer.py
   ```

2. Training the model:
   ```bash
   python3 trainer.py
   ```

Key training configuration parameters in `train_config.yaml`:
- Model architecture (layers, heads, embeddings)
- Training hyperparameters (batch size, learning rate, etc.)
- Checkpoint handling and model saving
- Hugging Face Hub integration

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the full pipeline:
   ```bash
   ./run_script.sh
   ```

3. For inference, use the trained model to generate sequences:
   ```python
   from modules.train_model import GPT2ModelTrainer
   trainer = GPT2ModelTrainer(config)
   generated = trainer.generate("1 2 3", max_length=100)
   ```

## Model Architecture
The implementation uses a GPT-2 style architecture with:
- Multiple transformer layers
- Multi-head self-attention
- Position embeddings
- Layer normalization
- Dropout regularization

## License
This project is licensed under the MIT License - see the LICENSE file for details.