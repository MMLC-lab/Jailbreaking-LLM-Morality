#!/usr/bin/env python3

import os
import re
import json
import torch
import argparse
from enum import Enum
import multiprocessing
import logging
from transformers import AutoTokenizer, Llama4ForConditionalGeneration
from huggingface_hub import login

login(token="YOUR_HF_TOKEN_HERE")

class TaskType(Enum):
    VAGUE = "vague"
    REVERSE = "reverse"
    FAKE = "fake"
    BIASED = "biased"

class Model(Enum):
    LGUARD_ORI = "lguard_ori"
    LGUARD_CUS = "lguard_cus"
    LGUARD_ORI_RESP = "lguard_ori_resp"
    LGUARD_CUS_RESP = "lguard_cus_resp"


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_response(response):
    """parse Llama Guard model response"""
    response = response.strip()
    
    # check if it contains unsafe indication
    if "unsafe" in response.lower():
        # try to extract specific unsafe category
        category_match = re.search(r'S(\d+)', response)
        category = category_match.group(0) if category_match else "unknown"
        
        return {
            "safety_label": "unsafe",
            "category": category,
            "full_response": response
        }
    else:
        return {
            "safety_label": "safe", 
            "category": None,
            "full_response": response
        }


def process_single_sample(item, model, tokenizer, device):
    """process a single sample"""
    try:
        # get input text
        prompt_text = item["full_prompt_or_instruction"]
        
        # tokenize
        tokenized_inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=4096
        ).to(device)
        
        # generate response
        with torch.no_grad():
            outputs = model.generate(
                **tokenized_inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # decode generated results (only take newly generated part)
        input_length = tokenized_inputs["input_ids"].shape[-1]
        generated_tokens = outputs[:, input_length:]
        response = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        # parse response
        parsed_data = parse_response(response)
        
        return {
            "response": response,
            "parsed_response": parsed_data
        }
        
    except Exception as e:
        logger.error(f"Error occurred while processing sample: {e}")
        return {
            "response": "error",
            "parsed_response": {
                "safety_label": "error",
                "category": None,
                "full_response": "error"
            }
        }


def main():
    # add command line argument parsing
    parser = argparse.ArgumentParser(description='Llama Guard reasoning...')
    parser.add_argument('--task_type', 
                       choices=[t.value for t in TaskType],
                       required=True,
                       help='task_type: vague/reverse/fake/biased')
    parser.add_argument('--model',
                       choices=[m.value for m in Model],
                       required=True,
                       help='model: lguard_ori/lguard_cus/lguard_ori_resp/lguard_cus_resp')
    
    args = parser.parse_args()
    task_type = args.task_type
    model_type = args.model
    
    # configure parameters
    model_id = "meta-llama/Llama-Guard-4-12B"
    
    # construct input file path based on task_type
    input_path = f"../../build_prompt/prompts/constructed_prompts_{task_type}_{model_type}"
    
    # create output directory
    output_dir = f"results_{task_type}_{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # check if input directory exists
    if not os.path.exists(input_path):
        logger.error(f"Input directory does not exist: {input_path}")
        return
    
    # load all JSON files
    json_files = [f for f in os.listdir(input_path) if f.endswith('.json')]
    
    if not json_files:
        logger.error(f"No JSON files in input directory: {input_path}")
        return
    
    logger.info(f"Loading model from {model_id}")

    try:
        # set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # initialize model
        model = Llama4ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="cuda",
            dtype=torch.bfloat16,
        )
        model.config.text_config.attention_chunk_size = 8192
        model.eval()  # set to evaluation mode
        
        logger.info("Model loaded and moved to device")
        logger.info(f"Starting to process {len(json_files)} samples...")
        
        for i, filename in enumerate(sorted(json_files)):
            # check if sample has already been inferred
            sample_id = filename.replace('.json', '')
            output_filename = f"{output_dir}/result_{sample_id}.json"
            
            if os.path.exists(output_filename):
                print(f"Skipping sample {i+1}/{len(json_files)}: ID={sample_id} (result file already exists)")
                continue
                
            file_path = os.path.join(input_path, filename)
            
            with open(file_path, "r", encoding="utf-8") as f:
                item = json.load(f)
            
            print(f"Processing sample {i+1}/{len(json_files)}: ID={sample_id}")
            
            # process single sample
            lguard_result = process_single_sample(item, model, tokenizer, device)
            
            # construct result
            result = {
                "sample_id": sample_id,
                "prompt_sample": item,
                "parsed_response": lguard_result,
            }
            
            # save result
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            safety_label = lguard_result["parsed_response"]["safety_label"]
            category = lguard_result["parsed_response"]["category"]
            print(f"Sample {sample_id} result saved to {output_filename} (Safety: {safety_label}, Category: {category})")
        
        print(f"Inference completed! All results saved to {output_dir}/ directory")
        
    except Exception as e:
        logger.error(f"Error during initialization or execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()