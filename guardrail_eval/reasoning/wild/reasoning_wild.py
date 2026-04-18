#!/usr/bin/env python3

import os
import re
import json
import torch
import argparse
from enum import Enum
import multiprocessing
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM


class TaskType(Enum):
    VAGUE = "vague"
    REVERSE = "reverse"
    FAKE = "fake"
    BIASED = "biased"

class Model(Enum):
    WILD_RESP = "wild_resp"
    WILD = "wild"


def parse_response(response):
    """Parse WildGuard model response"""
    response = response.strip()
    
    # WildGuard returns three judgment items in the format:
    # Harmful request: yes/no
    # Response refusal: yes/no  
    # Harmful response: yes/no
    
    # Extract the three judgment items
    harmful_request = None
    response_refusal = None
    harmful_response = None
    
    # Use regular expressions to extract each field
    harmful_request_match = re.search(r'Harmful request:\s*(yes|no)', response, re.IGNORECASE)
    if harmful_request_match:
        harmful_request = harmful_request_match.group(1).lower()
    
    response_refusal_match = re.search(r'Response refusal:\s*(yes|no)', response, re.IGNORECASE)
    if response_refusal_match:
        response_refusal = response_refusal_match.group(1).lower()
    
    harmful_response_match = re.search(r'Harmful response:\s*(yes|no)', response, re.IGNORECASE)
    if harmful_response_match:
        harmful_response = harmful_response_match.group(1).lower()
    
    
    return {
        "harmful_request": harmful_request,
        "response_refusal": response_refusal,
        "harmful_response": harmful_response,
        "full_response": response
    }


def process_single_sample(item, model, tokenizer, device):
    """Process a single sample"""
    try:
        # Get input text
        prompt_text = item["full_prompt_or_instruction"]
        
        # Tokenize
        tokenized_input = tokenizer([prompt_text], return_tensors='pt', add_special_tokens=False)
        for k in tokenized_input:
            tokenized_input[k] = tokenized_input[k].to(device)
        
        # Generate response
        with torch.no_grad():
            result = model.generate(**tokenized_input, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)
        
        # Decode response (only take newly generated part)
        response = tokenizer.decode(result[0][len(tokenized_input['input_ids'][0]):], skip_special_tokens=True)
        
        # Parse response
        parsed_data = parse_response(response)
        
        return {
            "response": response,
            "parsed_response": parsed_data
        }
        
    except Exception as e:
        return {
            "response": "error",
            "parsed_response": {
                "harmful_request": None,
                "response_refusal": None,
                "harmful_response": None,
                "full_response": "error"
            }
        }


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='WildGuard reasoning...')
    parser.add_argument('--task_type', 
                       choices=[t.value for t in TaskType],
                       required=True,
                       help='task_type: vague/reverse/fake/biased')
    parser.add_argument('--model', 
                       choices=[m.value for m in Model],
                       required=True,
                       help='model: wild/wild_resp')
    
    args = parser.parse_args()
    task_type = args.task_type
    model_type = args.model
    
    # Configure parameters
    model_id = "allenai/wildguard"
    
    # Construct input file path based on task_type and model_type
    input_path = f"../../build_prompt/prompts/constructed_prompts_{task_type}_{model_type}"
    
    # Create output directory
    output_dir = f"results_{task_type}_{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input directory exists
    if not os.path.exists(input_path):
        print(f"Input directory does not exist: {input_path}")
        return
    
    # Load all JSON files
    json_files = [f for f in os.listdir(input_path) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files in input directory: {input_path}")
        return

    print(f"Loading model from {model_id}")

    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
        model.eval()  # Set to evaluation mode
        
        print("Model loaded and moved to device")
        print(f"Starting to process {len(json_files)} samples...")
        
        for i, filename in enumerate(sorted(json_files)):
            # Check if the sample has already been inferred
            sample_id = filename.replace('.json', '')
            output_filename = f"{output_dir}/result_{sample_id}.json"
            
            if os.path.exists(output_filename):
                print(f"Skipping sample {i+1}/{len(json_files)}: ID={sample_id} (result file already exists)")
                continue
                
            file_path = os.path.join(input_path, filename)
            
            with open(file_path, "r", encoding="utf-8") as f:
                item = json.load(f)
            
            print(f"Processing sample {i+1}/{len(json_files)}: ID={sample_id}")
            
            # Process single sample
            wild_result = process_single_sample(item, model, tokenizer, device)
            
            # Construct result
            result = {
                "sample_id": sample_id,
                "prompt_sample": item,
                "response": wild_result["response"],
                "parsed_response": wild_result["parsed_response"],
            }
            
            # Save result
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"Sample {sample_id} result saved to {output_filename}")
        
        print(f"Inference completed! All results have been saved to the {output_dir}/ directory")
        
    except Exception as e:
        print(f"Error during initialization or execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()