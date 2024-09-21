import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import pandas as pd
import argparse
import yaml

# Load configuration from a YAML file
def load_config(config_file=None):
    if config_file:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            "model_name": "gemma-2-2B-it",  # Model updated to gemma-2-2B-it
            "learning_rate": 1e-5,
            "epochs_stage_1": 2,
            "epochs_stage_2": 3,
            "beta_kl": 0.1,
            "alpha": 1.0,
            "data_file": "SCoRe_dataset.csv"
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
        for example in data:

            # Format input using chat_template
            conversation = stage1_chat_format(example)
            
            # Convert conversation to a single string
            conversation_text = tokenizer.apply_chat_template(conversation, tokenize=False)
            
            inputs = tokenizer(conversation_text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs, labels=inputs['input_ids'])
            
            # Cross-entropy loss (first attempt)
            cross_entropy_loss = outputs.loss
            
            # Log probabilities and apply KL divergence loss
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
            with torch.no_grad():
                target_probs = F.softmax(logits, dim=-1)
            kl_loss = F.kl_div(log_probs, target_probs, reduction='batchmean')
            
            # Total loss combines cross-entropy and scaled KL divergence
            total_loss_value = cross_entropy_loss + beta_kl * kl_loss
            
            optimizer.zero_grad()
            total_loss_value.backward()
            optimizer.step()
            
            total_loss += total_loss_value.item()
        print(f"Stage I - Epoch {epoch+1}, Loss: {total_loss:.4f}")

def stage1_chat_format(example):
    return [
        {"role": "user", "content": example['question']},
        {"role": "assistant", "content": f"답변: {example.get('original_answer', '')}"}
    ]

def stage_two_training_with_reward_shaping(model, tokenizer, data, epochs=3, lr=1e-5, alpha=1.0):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        for example in data:
            # First attempt (y1): Generate the initial answer using chat_template
            conversation1 = stage2_chat_format(example)
            conversation_text1 = tokenizer.apply_chat_template(conversation1, tokenize=False)
            inputs1 = tokenizer(conversation_text1, return_tensors="pt", padding=True, truncation=True)
            inputs1 = {k: v.to(model.device) for k, v in inputs1.items()}
            
            # Generate output for the first attempt
            with torch.no_grad():
                outputs1 = model(**inputs1)
            
            # Second attempt (y2): Corrected answer
            conversation2 = chat_format(example)
            conversation_text2 = tokenizer.apply_chat_template(conversation2, tokenize=False)
            inputs2 = tokenizer(conversation_text2, return_tensors="pt", padding=True, truncation=True)
            inputs2 = {k: v.to(model.device) for k, v in inputs2.items()}
            
            # Forward pass with labels for loss calculation
            outputs2 = model(**inputs2, labels=inputs2['input_ids'])
            
            # Ensure we have a loss
            if outputs2.loss is None:
                print("Warning: Model output does not include loss. Using cross-entropy loss.")
                logits = outputs2.logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), inputs2['input_ids'].view(-1))
            else:
                loss = outputs2.loss
            
            # Compute reward based on self-correction
            generated_text1 = tokenizer.decode(outputs1.logits.argmax(dim=-1)[0], skip_special_tokens=True)
            generated_text2 = tokenizer.decode(outputs2.logits.argmax(dim=-1)[0], skip_special_tokens=True)
            reward = reward_function(generated_text1, generated_text2, example.get('correct_answer', ''))
            
            # Apply reward shaping
            shaped_loss = loss * reward
            
            optimizer.zero_grad()
            shaped_loss.backward()
            optimizer.step()
            
            total_loss += shaped_loss.item()
        print(f"Stage II - Epoch {epoch+1}, Total Loss: {total_loss:.4f}")

def stage2_chat_format(example):
    return [
        {"role": "user", "content": example['question']},
        {"role": "assistant", "content": f"첫 번째 답변: {example.get('original_answer', '')}"},
        {"role": "user", "content": "이 답변을 다시 한 번 검토해주세요."},
        {"role": "assistant", "content": "검토 후 답변: "}
    ]

def chat_format(example):
    return [
        {"role": "user", "content": example['question']},
        {"role": "assistant", "content": f"최종 답변: {example.get('correct_answer', '')}"}
    ]

# Main function to run the training process
def main(config_file=None):
    config = load_config(config_file)

    # Load model and tokenizer
    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 device_map="auto", 
                                                 attn_implementation='eager')

    # Load the dataset
    data_file_path = config["data_file"]
    df = pd.read_csv(data_file_path)

    # Prepare the data for Stage I and Stage II
    data_stage_one = df[["question", "original_answer"]].to_dict(orient="records")
    data_stage_two = df[["question", "original_answer", "correct_answer"]].to_dict(orient="records")

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
    tokenizer.save_pretrained("./trained_self_correcting_model")

# Run the main function (can use a config file or default)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file', default=None)
    args = parser.parse_args()

    main(args.config)