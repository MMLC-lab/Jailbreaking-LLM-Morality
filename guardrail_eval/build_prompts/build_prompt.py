import json
import os
import argparse
from enum import Enum
from typing import Dict, Any, List
# python build_prompt.py --task_type vague --model granite
only_instruction_models = ["mdjudge", "pguard", "granite_user", "granite_resp"]   # ~
prompt_instruction_resp_models = ["aegis_d_ori_resp", "aegis_p_ori_resp", "aegis_p_cus_resp", "aegis_d_cus_resp", "lguard_cus", "wild", "lguard_cus_resp", "lguard_ori_resp", "lguard_ori", "wild_resp", "shield_resp", "shield_single_resp"]  # ~
prompt_only_instruction_models = ["shield", "shield_single", "aegis_p_cus", "aegis_p_ori", "aegis_d_ori", "aegis_d_cus"]   # ~

class TaskType(Enum):
    VAGUE = "vague"
    REVERSE = "reverse"
    FAKE = "fake"
    BIASED = "biased"


class Model(Enum):
    AEGIS_D_ORI = "aegis_d_ori"
    AEGIS_P_CUS = "aegis_p_cus"
    AEGIS_P_ORI = "aegis_p_ori" # !
    AEGIS_D_CUS = "aegis_d_cus" # !
    AEGIS_D_CUS_RESP = "aegis_d_cus_resp" # !!
    AEGIS_P_CUS_RESP = "aegis_p_cus_resp" # !!
    AEGIS_D_ORI_RESP = "aegis_d_ori_resp" # !!!
    AEGIS_P_ORI_RESP = "aegis_p_ori_resp" # !!!
    GRANITE = "granite" # -
    LGUARD_CUS = "lguard_cus"
    LGUARD_CUS_RESP = "lguard_cus_resp" # !
    LGUARD_ORI = "lguard_ori" # !
    LGUARD_ORI_RESP = "lguard_ori_resp" # !
    WILD_RESP = "wild_resp" # !
    SHIELD_RESP = "shield_resp" # !
    SHIELD_SINGLE_RESP = "shield_single_resp" # !
    SHIELD_SINGLE = "shield_single" # !
    MDJUDGE = "mdjudge"
    PGUARD = "pguard"
    SHIELD = "shield"
    WILD = "wild"
    GRANITE_USER = "granite_user"
    GRANITE_RESP = "granite_resp"


def load_instruction_template() -> str:
    template_path = os.path.join(os.path.dirname(__file__), 'instruction_template.txt')
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_prompt_template(model: str) -> str:
    template_path = os.path.join(os.path.dirname(__file__), f'prompt_{model}.txt')
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_data(data_dir: str) -> List[Dict[str, Any]]:
    samples = []
    data_path = os.path.expanduser(data_dir)
    json_files = [f for f in os.listdir(data_path) if f.endswith('.json')]
    
    for file in sorted(json_files):
        file_path = os.path.join(data_path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = json.load(f)
            samples.append(sample)
    return samples



def build_one_prompt(response_dict: Dict[str, Any], sample: Dict[str, Any], task_type: str, model: str) -> List[Dict[str, Any]]:
    if task_type in ['vague', 'biased']:
        ground_truth = "immoral"
        attack_target = "moral"
    else:
        ground_truth = "moral"
        attack_target = "immoral"

    prompts = []
    
    norm = sample[f'{task_type}_norm']
    situation = sample['situation']
    intention = sample['intention']
    action = sample.get(f'{ground_truth}_action', sample.get('moralAction', ''))
    sample_id = sample.get('ID', sample.get('id', 'unknown_id')) # !

    instruction_template = load_instruction_template()
    instruction = instruction_template.format(norm, situation, intention, action)

    corresponding_resp = None

    if model not in only_instruction_models:
        prompt_template = load_prompt_template(model)
        if model in prompt_instruction_resp_models:
            if model.endswith('_resp'):   # !
                if sample_id not in response_dict:
                    print(f"Warning: No response found for sample ID={sample_id}. Skipping.")
                    answer = f"It is {attack_target}."
                else:
                    corresponding_resp = response_dict[sample_id]
                    if corresponding_resp.get('answer', '').strip() == '':
                        print(f"Warning: Empty answer for sample ID={sample_id}.")
                        answer = corresponding_resp['justification'] + f"It is {attack_target}."
                    else:
                        answer = corresponding_resp['justification'] + f"It is {corresponding_resp['answer']}."
                
                full_prompt = prompt_template.format(instruction=instruction, response=answer)  # ! ~
            else:
                full_prompt = prompt_template.format(instruction=instruction, attack_target=attack_target)  # ~
        else:   # shield/aegis
            full_prompt = prompt_template.format(instruction=instruction)

    prompt_record = {
        "task_type": task_type,
        "model": model,
        "sample": sample,
        "model_response": corresponding_resp,
        "full_prompt_or_instruction": full_prompt if model not in only_instruction_models else instruction,
    }
    prompts.append(prompt_record)
    return prompts

def build_prompts_from_samples(samples: List[Dict[str, Any]], task_type: str, model: str) -> List[Dict[str, Any]]:
    all_prompts = []

    # read response
    response_data = []
    file_path = f"../../generate_gemini_pro_response/gemini_{task_type}_norm_justification.jsonl"
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
    # end read response

    for sample in samples:
        sample_prompts = build_one_prompt(response_dict, sample, task_type, model)
        all_prompts.extend(sample_prompts)
    return all_prompts

def save_prompts(prompts: List[Dict[str, Any]], output_dir: str):
    for i, prompt in enumerate(prompts):
        sample_id = prompt["sample"].get('ID', prompt["sample"].get('id', f'sample_{i}'))
        filename = f'{sample_id}.json'
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(prompt, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description='constructing prompts...')
    parser.add_argument('--task_type', 
                       choices=[t.value for t in TaskType],
                       required=True,
                       help='task_type: vague/reverse/fake/biased')
    parser.add_argument('--model', 
                       choices=[m.value for m in Model],
                       required=True,
                       help='model: granite_user/granite_resp/aegis_d_ori_resp/aegis_p_ori_resp/aegis_p_cus_resp/aegis_d_cus_resp/aegis_d_ori/aegis_d_cus/aegis_p_cus/aegis_p_ori/lguard_cus/lguard_cus_resp/lguard_ori/lguard_ori_resp/mdjudge/pguard/shield/shield_resp/shield_single/shield_single_resp/wild/wild_resp')

    args = parser.parse_args()
    task_type = args.task_type
    model = args.model
    
    if task_type in ["vague", "reverse"]:
        data_dir = "../../data/value-ambiguity-2500"
    else:
        data_dir = "../../data/value-conflict-2800"

    samples = load_data(data_dir)
    prompts = build_prompts_from_samples(samples, task_type, model)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'prompts/constructed_prompts_{task_type}_{model}')
    os.makedirs(output_dir, exist_ok=True)
    save_prompts(prompts, output_dir)


if __name__ == "__main__":
    main()
