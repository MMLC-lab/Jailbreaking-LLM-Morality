# Jailbreaking LLM Morality
Code repo for ACL 2026 finding paper: __Jailbreaking Large Language Models with Morality Attacks__

### Data Construction
We first sample and select instances from original dataset [Moral Story](https://huggingface.co/datasets/demelin/moral_stories) and [ValuePrism](https://huggingface.co/datasets/allenai/ValuePrism). The selected and preprocessed data instances are under `./data` folder, namely `moral_story_select_2500.jsonl` and `valueprism_select_2800.jsonl`.

With [Google AI Studio](https://aistudio.google.com/welcome?utm_source=PMAX&utm_medium=display&utm_campaign=FY25-global-DR-pmax-1710442&utm_content=pmax&gclsrc=aw.ds&gad_source=1&gad_campaignid=21766228245&gbraid=0AAAAACn9t64BWlcRKTwz3O1y5Cl59CBLw&gclid=CjwKCAjwi4PHBhA-EiwAnjTHue-_oJN6io2Hnv1HmHJIwZ5WkNDNTXodD9GPgJooor8qlJ4iXwpAIBoCwWsQAvD_BwE), we prompt `Gemini-Pro-2.5` through google API to construct designed morality instance. 

For Value Ambiguity, we use `moral_story_select_2500.jsonl` and run:
```
python prompt_moralstory.py
```

For Value Conflict, we use `valueprism_select_2800.jsonl` and run:
```
python prompt_valueprism.py
```
After prompting, we filter the instances and get final constructed dataset.

`value-ambiguity-2500.zip`: unzip the folder and get value ambiguity instance with reverse norm and vague norm.

`value-ambiguity-2500-analyze.zip`: unzip the folder and get attribute annotation of each value ambiguity instance.

`value-conflict-2800.zip`: unzip the folder and get value conflict instance with fake norm and biased norm.

`value-conclict-2800-analyze.zip`: unzip the folder and get attribute annotation of each value conflict instance.

### Prompting LLMs
For prompting LLM to get judgment and justification under folder './prompting_LLM', we first sample 1/10 of the full dataset. The sampled instance ids are in `moral_story_select.txt` and `valueprism_select.txt` respectively. 

We give the example of prompting deepseek as an example, the designed prompt for attack RN, VN, FN, and BN are the sample for the other LLMs (such as GPT-4.1-mini, Llama-4-maverick, etc).

```
python deepseek_FN.py
```
This script gets the result of prompting deepseek with FN attack. The other three types of attacks are in a similar way. 

### Prompting Gemini-Pro-2.5 as LLM response
Under the folder `./generate_gemini_pro_response`, we prompt Gemini-2.5-pro to get the model response which will be used in the prompting guardrail model section.

Unlike previously prompting Gemini-2.5-pro to get judgment and justification, we presumably to judgement is exactly the wrong answer for each type of attack, and make the model to generate justification. This ensures the unification of model response.

```
python gemini_RN.py
```
This script gets the `gemini_reverse_norm_justification.jsonl` result for justification of reverse attack. For other three types of attacks, the results are acquired in a similar way. All the responses are listed under the folder `./generate_gemini_pro_response`.

### Prompting Guardrail Models
For prompting guardrail models, we have two lines of experiments: prompting with user input and prompting with model response. 

#### Usage
- Step 1: Initialize Environment
  - Create and activate a Conda environment with Python 3.13:
    - ```bash
        conda create -n <env_name> python=3.13
        conda activate <env_name>
        git clone <this_repo>
        cd </your_path_of_repo>
      ```
- Step 2: Dependency Versions
  - Experiments conducted on cuda 12.2 platform
    - ```bash
        pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
        pip install vllm
        pip install transformers==4.56.2
        pip install peft==0.17.1
      ```
- Step 3: Build Prompts for Guardrail Models
  - Navigate to the prompt building directory and execute the prompt construction script:
    - ```bash
        cd ./build_prompt
        python build_prompt.py --task_type <task_type> --model <model>
      ```
  - where parameter "`task_type`" and "`model`" are one of the values as below:
    - | `task_type` | intro |
      |------|------|
      | vague | To induce the model to wrongly judge moral behaviors that conform to the golden norm as "immoral". |
      | reversed | To induce the model to wrongly judge moral behaviors that violate the golden norm as "moral". |
      | biased | To induce the model to wrongly judge the balanced behavior that takes into account both values as "immoral". |
      | fake | To induce the model to wrongly identify biased behaviors that only consider a single value as "moral". |
    - |`guardrail_dir`| model ref | `model` | EXP_TYPE |
      |--|---|------|------|
      |aegis|[Aegis-Defensive-1.0](https://huggingface.co/nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0)| aegis_d_cus | CP, U |
      ||| aegis_d_ori | OP, U |
      ||| aegis_d_cus_resp | CP, R |
      ||| aegis_d_ori_resp | OP, R |
      ||[Aegis-Permissive-1.0](https://huggingface.co/nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0)| aegis_p_cus | CP, U |
      ||| aegis_p_ori | OP, U |
      ||| aegis_p_cus_resp | CP, R |
      ||| aegis_p_ori_resp | OP, R |
      |lguard|[Llama-Guard-4-12B](https://huggingface.co/meta-llama/Llama-Guard-4-12B)| lguard_cus | CP, U |
      ||| lguard_ori | OP, U |
      ||| lguard_cus_resp | CP, R |
      ||| lguard_ori_resp | OP, R |
      |shield|[shieldgemma-9b](https://huggingface.co/google/shieldgemma-9b)| shield | CP, U, M |
      |||shield_resp| CP, R, M|
      |||shield_single| CP, U, S|
      |||shield_single_resp| CP, R, S|
      |wild|[wildguard](https://huggingface.co/allenai/wildguard)| wild_resp | U+R (wildguard can evaluate both at once) |
      |pguard|[PromptGuard-86M](https://huggingface.co/meta-llama/Prompt-Guard-86M)| pguard | U |
      |mdjudge|[MD-Judge-v0_2-internlm2_7b](https://huggingface.co/OpenSafetyLab/MD-Judge-v0_2-internlm2_7b)| mdjudge | R |
      |granite|[granite-guardian-3.3-8b](https://huggingface.co/ibm-granite/granite-guardian-3.3-8b)| granite_user | U |
      ||| granite_resp | R |

  - generated prompts are final prompts being fed towards guardrails, which are stored in `build_prompt/prompts/`
- Step 4: Reasoning with Guardrails
  - change path to reasoning scripts and conduct reasoning
    - ```bash
        cd ../reasoning/<guardrail_dir>
        python reasoning_<guardrail_dir>.py --task_type <task_type> --model <model>
      ```
    - where `guardrail_dir`, `model` and `task_type` are one of the values as above
    - the reasoning results will be stored in `<guardrail_dir>\results_<task_type>_<model>\`
- Step 5: Analyse the results
  - Copy the experimental result directory of each task type along with corresponding evaluation script to the analysis directory
    - ```bash
        cd </your_path_of_repo>/analysis
        cp -r "</your_path_of_repo>/reasoning/<guardrail_dir>/results_<task_type>_<model>/" "</your_path_of_repo>/analysis"
        cp -r "</your_path_of_repo>/eval_scripts/<guardrail_dir>.ipynb" "</your_path_of_repo>/analysis"
      ```
    - then run cells of ipynb doc to see the results
