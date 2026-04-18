#!/usr/bin/env python3

import os
# Disable torch.compile to avoid compatibility issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import argparse
from enum import Enum
import multiprocessing


class TaskType(Enum):
    VAGUE = "vague"
    REVERSE = "reverse"
    FAKE = "fake"
    BIASED = "biased"

class Model(Enum):
    SHIELD_SINGLE_RESP = "shield_single_resp"
    SHIELD_SINGLE = "shield_single"
    SHIELD = "shield"
    SHIELD_RESP = "shield_resp"


# ShieldGemma 9B
MODEL_ID = "google/shieldgemma-9b"
LOCAL_MODEL_DIR = os.path.expanduser("~/models/shieldgemma-9b")


def process_single_sample(item, model, tokenizer):
    # Use Shield prompt
    prompt_text = item["full_prompt_or_instruction"]

    # Generate full textual response
    tokenized_inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding="longest", 
        truncation=True,
        max_length=4096
    ).to(model.device)

    outputs = model.generate(
        **tokenized_inputs,
        max_new_tokens=256,
        do_sample=False,
        use_cache=False,  # Disable cache to avoid compatibility issues
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode generated outputs (only the newly generated tokens)
    input_length = tokenized_inputs["input_ids"].shape[-1]
    generated_tokens = outputs[:, input_length:]
    response = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    # Get stripped raw_response
    raw_response_stripped = response[0].strip() if response else ""
    
    return {
        "response": raw_response_stripped,
    }
    


def main():
    # Add command-line argument parsing
    parser = argparse.ArgumentParser(description='Shield reasoning...')
    parser.add_argument('--task_type', 
                       choices=[t.value for t in TaskType],
                       required=True,
                       help='task_type: vague/reverse/fake/biased')
    parser.add_argument('--model',
                       choices=[m.value for m in Model],
                       required=True,
                       help='model: shield_single_resp/shield_single/shield/shield_resp')
    
    args = parser.parse_args()
    task_type = args.task_type
    model_type = args.model
    
    # Build input path based on task_type
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
        print(f"No JSON files found in input directory: {input_path}")
        return
    
    # Prefer local model, otherwise use remote model
    if os.path.exists(LOCAL_MODEL_DIR):
        print(f"Using local model: {LOCAL_MODEL_DIR}")
        model_path = LOCAL_MODEL_DIR
    else:
        print(f"Using remote model: {MODEL_ID}")
        model_path = MODEL_ID
    
    print("Initializing ShieldGemma model...")
    
    try:
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize model - disable torch.compile to avoid compatibility issues
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto", 
            dtype=torch.bfloat16,
        )
        
        # Disable torch.compile
        model._orig_mod = model
        print(f"Starting to process {len(json_files)} samples...")
                
        for i, filename in enumerate(sorted(json_files)):
            # 检查是否已经推理过该样本
            sample_id = filename.replace('.json', '')
            output_filename = f"{output_dir}/result_{sample_id}.json"
            
            if os.path.exists(output_filename):
                print(f"Skipping sample {i+1}/{len(json_files)}: ID={sample_id} (result file already exists)")
                continue
                
            file_path = os.path.join(input_path, filename)
            
            with open(file_path, "r", encoding="utf-8") as f:
                item = json.load(f)
            
            print(f"Processing sample {i+1}/{len(json_files)}: ID={sample_id}")

            # Process a single sample
            shield_result = process_single_sample(item, model, tokenizer)

            # Build result
            result = {
                "sample_id": sample_id,
                "prompt_sample": item,
                "parsed_response": shield_result
            }

            # Save results
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"Sample {sample_id} results saved to {output_filename}")
        
        print(f"Inference complete! All results saved to the {output_dir}/ directory")
        
    except Exception as e:
        print(f"Error during initialization or runtime: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
