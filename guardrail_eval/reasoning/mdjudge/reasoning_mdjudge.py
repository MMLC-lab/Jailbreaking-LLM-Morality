from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import argparse
import os
from enum import Enum


class TaskType(Enum):
    VAGUE = "vague"
    REVERSE = "reverse"
    FAKE = "fake"
    BIASED = "biased"


def parse_result(response: str) -> dict:
    """
    Parse MD-Judge model response
    """
    import re
    
    result_dict = {
        "parsed_result": "",
        "judgement": "",
        "score": None,
        "unsafe_category": ""
    }
    
    # Find content after [RESULT]:
    result_match = re.search(r'\[RESULT\]:\s*(.+)', response, re.IGNORECASE)
    if result_match:
        result_content = result_match.group(1).strip()
        result_dict["parsed_result"] = result_content
        
        # Extract judgement (safe/unsafe) from result_content
        if result_content.lower().startswith('safe'):
            result_dict["judgement"] = "safe"
        elif result_content.lower().startswith('unsafe'):
            result_dict["judgement"] = "unsafe"
        
        # Extract score (number)
        score_match = re.search(r'score\s+(\d+)', result_content, re.IGNORECASE)
        if score_match:
            result_dict["score"] = int(score_match.group(1))
    
    # Find content after [UNSAFE CATEGORY]:
    unsafe_category_match = re.search(r'\[UNSAFE CATEGORY\]:\s*(.+)', response, re.IGNORECASE)
    if unsafe_category_match:
        result_dict["unsafe_category"] = unsafe_category_match.group(1).strip()
    
    # If [RESULT]: tag is not found, use the entire response as parsed_result
    if not result_dict["parsed_result"]:
        result_dict["parsed_result"] = response.strip()
    
    return result_dict


def main():
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='MD-Judge reasoning...')
    parser.add_argument('--task_type', 
                       choices=[t.value for t in TaskType],
                       required=True,
                       help='task_type: vague/reverse/fake/biased')
    parser.add_argument('--model', 
                        choices=['mdjudge'],
                        required=False,
                        type=str,
                        help='model: mdjudge')
    
    args = parser.parse_args()
    task_type = args.task_type
    
    model_path = "OpenSafetyLab/MD-Judge-v0_2-internlm2_7b"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Construct input file path based on task_type
    input_path = f"../../build_prompt/prompts/constructed_prompts_{task_type}_mdjudge"

    # Load all JSON files
    json_files = [f for f in os.listdir(input_path) if f.endswith('.json')]

    # Create output directory
    output_dir = f"results_{task_type}_mdjudge"
    os.makedirs(output_dir, exist_ok=True)

    # Read response
    response_data = []
    file_path = f"./llm_resp/gemini_{task_type}_norm_justification.jsonl"
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    response_data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning for line: {line.strip()}")
    response_dict = {}
    for item in response_data:
        key = item['ID']
        response_dict[key] = item


    for i, filename in enumerate(sorted(json_files)):
        file_path = os.path.join(input_path, filename)
        
        with open(file_path, "r", encoding="utf-8") as f:
            item = json.load(f)
        
        print(f"Processing sample {i+1}/{len(json_files)}: ID={item.get('sample', {}).get('id', filename)}")
        
        # Check if the sample has already been inferred
        sample_id = item.get('sample', {}).get('id', filename.replace('.json', ''))
        output_filename = f"{output_dir}/result_{sample_id}.json"
        if os.path.exists(output_filename):
            print(f"Skip {i+1}/{len(json_files)}: ID={sample_id}. Already exists.")
            continue

        question = item["full_prompt_or_instruction"]

        if task_type in ['vague', 'biased']:
            ground_truth = "immoral"
            attack_target = "moral"
        else:
            ground_truth = "moral"
            attack_target = "immoral"
        
        sample_flag = True

        if sample_id not in response_dict:
            print(f"Warning: No response found for sample ID={sample_id}. Skipping.")
            answer = f"It is {attack_target}."
            sample_flag = False
        else:
            corresponding_resp = response_dict[sample_id]
            if corresponding_resp.get('answer', '').strip() == '':
                print(f"Warning: Empty answer for sample ID={sample_id}.")
                answer = corresponding_resp['justification'] + f"It is {attack_target}."
            else:
                answer = corresponding_resp['justification'] + f"It is {corresponding_resp['answer']}."
        
        # Apply MD-Judge's chat template
        prompt = tokenizer.apply_chat_template([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ], tokenize=False, add_generation_prompt=True)
        
        # Check and truncate input length
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, 
                          max_length=3000, truncation=True)
        
        # Move inputs to the device where the model is located
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        print(f"Input length: {len(inputs['input_ids'][0])}")
        
        # Clear GPU cache to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, 
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Clear GPU cache again to prevent memory leaks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # Decode the generated results (only take the newly generated part)
        input_length = len(inputs["input_ids"][0])
        generated_tokens = outputs[0].cpu().tolist()[input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        
        # Parse various fields from the response
        parsed_data = parse_result(response)
        
        result = {
            "ID": item.get('sample', {}).get('id', filename),
            "prompt_sample": item,
            "llm_resp": corresponding_resp if sample_flag else {},
            "raw_response": response,
            "parsed_response": parsed_data,
        }
        
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"Sample {sample_id} result saved to {output_filename}")

    print(f"Inference completed! All results have been saved to the {output_dir}/ directory")


if __name__ == "__main__":
    main()