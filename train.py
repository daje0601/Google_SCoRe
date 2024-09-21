import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import pandas as pd
import argparse
import json

# Load configuration from a JSON file or arguments
import yaml

# Load configuration from a YAML file
def load_config(config_file=None):
    if config_file:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            "model_name": "gpt2",
            "learning_rate": 1e-5,
            "epochs_stage_1": 2,
            "epochs_stage_2": 3,
            "beta_kl": 0.1,
            "alpha": 1.0,
            "data_file": "south_korea_qa_dataset_korean_various_topics.csv"
        }
    return config


# Reward function for self-correction
def reward_function(original_answer, corrected_answer, correct_answer):
    if corrected_answer == correct_answer:  # Fully correct answer
        return 1.0
    elif corrected_answer == original_answer:  # No improvement from original answer
        return -1.0
    else:
        return 0.5  # Partial improvement, better than original but still incorrect

# Stage I: Train initial model to generate first attempt (y1) and prevent mode collapse
def stage_one_initialization(model, tokenizer, data, epochs=2, lr=1e-5, beta_kl=0.1):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        for question, original_answer in data:
            # Prepare input data
            inputs = tokenizer(question, return_tensors="pt")
            outputs = model(**inputs, labels=inputs['input_ids'])
            
            # Cross-entropy loss (first attempt)
            cross_entropy_loss = outputs.loss
            
            # Log probabilities and apply KL divergence loss
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
            with torch.no_grad():
                target_probs = F.softmax(inputs['input_ids'], dim=-1)

            kl_loss = F.kl_div(log_probs, target_probs, reduction='batchmean')
            
            # Total loss combines cross-entropy and scaled KL divergence
            total_loss_value = cross_entropy_loss + beta_kl * kl_loss
            
            optimizer.zero_grad()
            total_loss_value.backward()
            optimizer.step()
            
            total_loss += total_loss_value.item()
        print(f"Stage I - Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Stage II: Multi-turn RL with reward shaping for self-correction
def stage_two_training_with_reward_shaping(model, tokenizer, data, epochs=3, lr=1e-5, alpha=1.0):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        for question, original_answer, corrected_answer, correct_answer in data:
            # First attempt (y1): Generate the initial answer
            inputs = tokenizer(question, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits

            # Second attempt (y2): Generate the corrected answer
            corrected_inputs = tokenizer(corrected_answer, return_tensors="pt")
            corrected_outputs = model(**corrected_inputs)

            # Compute reward based on self-correction
            reward = reward_function(original_answer, corrected_answer, correct_answer)

            # Loss for second attempt (reward shaping)
            cross_entropy_loss = nn.CrossEntropyLoss()(corrected_outputs.logits.view(-1, model.config.vocab_size), corrected_inputs['input_ids'].view(-1))
            shaped_loss = reward * cross_entropy_loss

            optimizer.zero_grad()
            shaped_loss.backward()
            optimizer.step()

            total_loss += shaped_loss.item()
        print(f"Stage II - Epoch {epoch+1}, Total Loss: {total_loss:.4f}")

# Main function to run the training process
def main(config_file=None):
    config = load_config(config_file)

    # Load model and tokenizer
    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load the dataset
    data_file_path = config["data_file"]
    df = pd.read_csv(data_file_path)

    # Prepare the data for Stage I and Stage II
    data_stage_one = df[["question", "original_answer"]].values.tolist()
    data_stage_two = df[["question", "original_answer", "correct_answer"]].values.tolist()

    # Stage I training (Initialization)
    stage_one_initialization(
        model, tokenizer, data_stage_one, 
        epochs=config["epochs_stage_1"], 
        lr=config["learning_rate"], 
        beta_kl=config["beta_kl"]
    )

    # Stage II training (Self-correction)
    stage_two_training_with_reward_shaping(
        model, tokenizer, data_stage_two, 
        epochs=config["epochs_stage_2"], 
        lr=config["learning_rate"], 
        alpha=config["alpha"]
    )

    # Save the trained model
    model.save_pretrained("./trained_self_correcting_model")

# Run the main function (can use a config file or default)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file', default=None)
    args = parser.parse_args()

    main(args.config)
