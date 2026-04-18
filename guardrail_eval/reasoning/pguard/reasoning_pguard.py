#!/usr/bin/env python3

import json
import os
import time
import torch
from typing import List, Dict, Any
from pathlib import Path
import logging
from datetime import datetime
import argparse
from enum import Enum
import multiprocessing

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification
)


class TaskType(Enum):
    VAGUE = "vague"
    REVERSE = "reverse"
    FAKE = "fake"
    BIASED = "biased"


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def process_single_sample(item, model, tokenizer, device):
    """process a single sample with PGuard model"""
    try:
        # Obtain the full prompt or instruction
        prompt_text = item["full_prompt_or_instruction"]
        
        # Tokenize
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference - classification task
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Obtain prediction results
            predicted_class_id = logits.argmax().item()
            probabilities = torch.softmax(logits, dim=-1).squeeze()
        
        # Obtain model configuration
        model_config = model.config
        model_config.id2label = {0: "safe", 1: "jailbreak"}
        # Obtain label name
        predicted_label = model_config.id2label.get(predicted_class_id, f"LABEL_{predicted_class_id}")
        confidence = float(probabilities[predicted_class_id].cpu())
        
        return {
            "predicted_class_id": predicted_class_id,
            "predicted_label": predicted_label,
            "confidence": confidence,
        }
        
    except Exception as e:
        logger.error(f"Error processing sample: {e}")
        return {
            "predicted_class_id": -1,
            "predicted_label": "error",
            "confidence": 0.0,
        }


def main():
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='PGuard reasoning...')
    parser.add_argument('--task_type', 
                       choices=[t.value for t in TaskType],
                       required=True,
                       help='task_type: vague/reverse/fake/biased')
    parser.add_argument('--model',
                       type=str,
                       choices=['pguard'],
                       required=False,
                       help='model: pguard')
    
    args = parser.parse_args()
    task_type = args.task_type
    
    # Configure parameters - prefer local path, fallback to HuggingFace model if not exists
    LOCAL_MODEL_PATH = "your/local/path/to/pguard-model"
    ONLINE_MODEL_PATH = "meta-llama/Prompt-Guard-2-86M"
    
    # Check if local model exists
    if os.path.exists(LOCAL_MODEL_PATH):
        MODEL_PATH = LOCAL_MODEL_PATH
        logger.info(f"Using local model: {MODEL_PATH}")
    else:
        MODEL_PATH = ONLINE_MODEL_PATH
        logger.info(f"Local model not found, using online model: {MODEL_PATH}")
    
    # Construct input file path based on task_type
    input_path = f"../../build_prompt/prompts/constructed_prompts_{task_type}_pguard"
    
    # Create output directory
    output_dir = f"results_{task_type}_pguard"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input directory exists
    if not os.path.exists(input_path):
        logger.error(f"Input directory does not exist: {input_path}")
        return
    
    # Load all JSON files
    json_files = [f for f in os.listdir(input_path) if f.endswith('.json')]
    
    if not json_files:
        logger.error(f"No JSON files in input directory: {input_path}")
        return
    
    logger.info(f"Loading model from {MODEL_PATH}")
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        
        # Ensure pad_token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure model loading parameters
        model_kwargs = {
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
        }
        
        # Load classification model
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            **model_kwargs
        )
        
        # Move model to device
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        logger.info("Model loaded and moved to device")
        
        logger.info(f"Starting to process {len(json_files)} samples...")
        
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
            
            print(f"处理样本 {i+1}/{len(json_files)}: ID={sample_id}")
            
            # Process single sample
            pguard_result = process_single_sample(item, model, tokenizer, device)
            
            # Construct result
            result = {
                "sample_id": sample_id,
                "prompt_sample": item,
                "parsed_response": pguard_result
            }
            
            # Save result
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
        print(f"Inference completed! All results have been saved to the {output_dir}/ directory")
        
    except Exception as e:
        logger.error(f"Error during initialization or execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
