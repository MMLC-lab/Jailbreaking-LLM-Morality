import os, re
from transformers import AutoTokenizer
import torch
import json
import argparse
from enum import Enum
from vllm import LLM, SamplingParams


class TaskType(Enum):
    VAGUE = "vague"
    REVERSE = "reverse"
    FAKE = "fake"
    BIASED = "biased"


def parse_response(response):
    """Parse the response from the granite model to extract score, thought, and full response"""
    score_match = re.findall(r'<score>(.*?)</score>', response, re.DOTALL)
    score = score_match[-1].strip() if score_match else None
    
    think_match = re.findall(r'<think>(.*?)</think>', response, re.DOTALL)
    think = think_match[-1].strip() if think_match else None
    
    return {
        "score": score,
        "think": think,
        "full_response": response.strip()
    }


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the GPU device to use
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Granite reasoning...')
    parser.add_argument('--task_type', 
                       choices=[t.value for t in TaskType],
                       required=True,
                       help='task_type: vague/reverse/fake/biased')
    parser.add_argument('--model', 
                       type=str,
                       required=True,
                       choices=["granite_user", "granite_resp"],
                       help='model type: granite_user/granite_resp')
    
    args = parser.parse_args()
    task_type = args.task_type
    model_type = args.model
    
    model_name = "ibm-granite/granite-guardian-3.3-8b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

    # Add single GPU configuration, disable torch.compile to avoid compatibility issues
    model = LLM(
        model=model_name, 
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
        disable_log_stats=True,
        enforce_eager=True,  # Disable torch.compile
        enable_prefix_caching=False,  # Disable prefix caching
        disable_custom_all_reduce=True  # Disable custom all_reduce
    )

    # Construct input file path based on task_type
    input_path = f"../../build_prompt/prompts/constructed_prompts_{task_type}_{model_type}"

    # Load all JSON files
    json_files = [f for f in os.listdir(input_path) if f.endswith('.json')]

    # Create output directory
    output_dir = f"results_{task_type}_{model_type}"
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
        
        sample_id = item.get('sample', {}).get('id', filename.replace('.json', ''))
        print(f"Processing {i+1}/{len(json_files)}: ID={sample_id}")
        
        # check if already processed
        output_filename = f"{output_dir}/result_{sample_id}.json"
        if os.path.exists(output_filename):
            print(f"Skip {i+1}/{len(json_files)}: ID={sample_id}. Already exists.")
            continue


        # ================== get attack target and answer
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
        # ================== get attack target and answer

        user_text = item["full_prompt_or_instruction"]
        messages = [{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}] if model_type == "granite_resp" else [{"role": "user", "content": user_text}]
        guardian_config = {"criteria_id": "harm_engagement"} if model_type == "granite_resp" else {"criteria_id": "jailbreak"}
        print(type(tokenizer))
        chat = tokenizer.apply_chat_template(messages, guardian_config=guardian_config, think=True, tokenize=False, add_generation_prompt=True)
        
        print(f"Chat template generated for sample {i+1}")
        
        output = model.generate(chat, sampling_params, use_tqdm=False)
        response = output[0].outputs[0].text.strip()
        
        # Parse response
        parsed_data = parse_response(response)
        
        result = {
            "sample_id": item.get('sample', {}).get('id', filename),
            "prompt_sample": item,
            "llm_resp": corresponding_resp if sample_flag else {},
            "response": response,
            "parsed_response": parsed_data,
        }
        
        # Save each sample to a separate file
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"Sample {sample_id} results saved to {output_filename}")
    print(f"Inference completed! All results have been saved to the {output_dir}/ directory")


if __name__ == "__main__":
    main()