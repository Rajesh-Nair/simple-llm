# Simple LLM Training with GPT-2 Architecture

This repository demonstrates how to train a Language Learning Model (LLM) from scratch using the GPT-2 architecture. The model is trained on numerical sequences to learn and predict patterns.

## Overview
The project implements a complete ML pipeline including:
- Synthetic dataset generation for number sequences
- Custom tokenizer training
- Model training using GPT-2 architecture
- Inference capabilities

## Dataset Generator
The data generator creates synthetic number sequences based on patterns. To generate training data:

1. Configure `data_config.yaml` with desired parameters:
   - Sequence length and pattern rules
   - Number of sequences to generate
   - Output file path and format
   - Value ranges and constraints

2. Run the data generator:
   ```bash
   python data_generator.py
   ```

## Tokenizer
The project uses a custom BPE (Byte Pair Encoding) tokenizer trained specifically for numerical sequences:

1. The tokenizer is trained on the generated sequences
2. Configuration is managed through `train_config.yaml`
3. Supports vocabulary size customization
4. Handles special tokens (BOS, EOS, padding)

To train the tokenizer: