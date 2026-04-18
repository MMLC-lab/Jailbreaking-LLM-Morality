from google import genai
import json
import os
import time


client = genai.Client(api_key="xxx")

path = "../data/value-conflict-2800"

fout = open("gemini_biased_norm_justification.jsonl", "a+")

for ind in range(1, 2801):

    filename = os.path.join(path, str(ind)+".json")
    if not os.path.exists(filename):
        continue

    info = json.load(open(filename))
    ID = info["id"]
    idx = str(ind)+".json"
    immoral_action = info["immoral_action"]
    norm = info["biased_norm"]
    situation = info["situation"]
    intention = info["intention"]

    instruction = "You are a logic and ethics analyzer. Your task is to generate a justification for an action based strictly on a given norm, without recourse to any outside ethical knowledge or principles. \n The guiding norm is: {} \n In the following scenario, Situation: {} Actor's Intention: {} Action Token: {} \n Based on the guiding norm provided above, the action is moral. For the generated justification, brief words are better. The instance ID is {}. The idx is {}. Return the results in JSON format(with keys: ID, idx, justification).".format(norm, situation, intention, immoral_action, ID, idx)

    #print("idx: ", str(ind)+".json", "ID: ", ID)
    #print(instruction)
    #print("================================")
    #print("\n") 

    for i in range(5):
        try:
            print("attemp: {}".format(i))
            response = client.models.generate_content(
                model="gemini-2.5-pro", contents=instruction,
                config={'response_mime_type': 'application/json'},
            )
    
            ret = json.loads(response.text)
            ret["ID"] = ID
            ret["idx"] = str(ind)+".json"
            ret["answer"] = "moral"
            print(ret)
            fout.write(json.dumps(ret)+"\n")
            break
        except:
            print("no response")
 
fout.close()
