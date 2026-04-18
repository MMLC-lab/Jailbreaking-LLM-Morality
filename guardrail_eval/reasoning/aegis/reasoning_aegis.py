#!/usr/bin/env python3

import json
import warnings
import os
import argparse
from enum import Enum
import multiprocessing
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel

class TaskType(Enum):
    VAGUE = "vague"
    REVERSE = "reverse"
    FAKE = "fake"
    BIASED = "biased"

class Model(Enum):
    AEGIS_P_ORI = "aegis_p_ori"
    AEGIS_P_CUS = "aegis_p_cus"
    AEGIS_D_ORI = "aegis_d_ori"
    AEGIS_D_CUS = "aegis_d_cus"
    AEGIS_P_ORI_RESP = "aegis_p_ori_resp"
    AEGIS_P_CUS_RESP = "aegis_p_cus_resp"
    AEGIS_D_ORI_RESP = "aegis_d_ori_resp"
    AEGIS_D_CUS_RESP = "aegis_d_cus_resp"


warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

TOKEN = "xxx"
AEGIS_VARIANT_P = "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0"
AEGIS_VARIANT_D = "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0"


def classify_single_sample(text, model, tokenizer, device):
    try:
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        lines = response.strip().split('\n')
        safety_assessment = lines[0].strip().lower()
        violated_categories = []
        
        if len(lines) > 1 and safety_assessment == 'unsafe':
            violated_categories = [cat.strip() for cat in lines[1].split(',')]
        
        return {
            'assessment': safety_assessment,
            'violated_categories': violated_categories,
            'raw_response': response.strip()
        }

    except Exception as e:
        print(f"err: {e}")
        return "error"


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # command parsing
    parser = argparse.ArgumentParser(description='Aegis reasoning...')
    parser.add_argument('--task_type', 
                       choices=[t.value for t in TaskType],
                       required=True,
                       help='task_type: vague/reverse/fake/biased')
    parser.add_argument('--model', required=True, help='model name', choices=[m.value for m in Model])
    args = parser.parse_args()
    task_type = args.task_type
    model_type = args.model
    
    # input path
    input_path = f"../../build_prompt/prompts/constructed_prompts_{task_type}_{model_type}"
    
    # output dir
    output_dir = f"results_{task_type}_{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # check input path
    if not os.path.exists(input_path):
        print(f"Input path not found: {input_path}")
        return
    
    # load all JSON files
    json_files = [f for f in os.listdir(input_path) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in input directory: {input_path}")
        return
    
    print("Initializing Aegis Classifier...")
    
    try:
        model_id = "meta-llama/LlamaGuard-7b"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        path_to_adapter_weights = AEGIS_VARIANT_P if '_p_' in model_type else AEGIS_VARIANT_D
        lg_defensive_model = PeftModel.from_pretrained(model, path_to_adapter_weights)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lg_defensive_model = lg_defensive_model.to(device)
        lg_defensive_model.eval()
        
        print(f"Start {len(json_files)} samples...")
        
        for i, filename in enumerate(sorted(json_files)):
            # check if the sample has already been inferred
            sample_id = filename.replace('.json', '')
            output_filename = f"{output_dir}/result_{sample_id}.json"
            
            if os.path.exists(output_filename):
                print(f"Skip sample {i+1}/{len(json_files)}: ID={sample_id} (result file already exists)")
                continue
                
            file_path = os.path.join(input_path, filename)
            
            with open(file_path, "r", encoding="utf-8") as f:
                item = json.load(f)
            
            print(f"Processing sample {i+1}/{len(json_files)}: ID={sample_id}")
            
            # Get the text to classify
            user_text = item["full_prompt_or_instruction"]
            
            # Perform classification
            prediction = classify_single_sample(user_text, lg_defensive_model, tokenizer, device)
            
            # Construct result
            result = {
                "sample_id": sample_id,
                "prompt_sample": item,
                "model": "aegis_d",
                "parsed_response": prediction
            }
            
            # Save result
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"Sample {sample_id} result saved to {output_filename} (Prediction: {prediction})")
        
        print(f"Inference completed! All results have been saved to the {output_dir}/ directory")
        
    except Exception as e:
        print(f"Error during initialization or execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
