from google import genai
import json
import os
import time

client = genai.Client(api_key="xxx")

j = 0
save_dir = ".moral-stories-2500-full/"

for line in open("./moral_story_select_2500.jsonl"):
    j += 1
    info = json.loads(line.strip())
    ID = info["ID"]
    norm = info["norm"]
    situation = info["situation"]
    intention = info["intention"]
    moral_action = info["moral_action"]
    moral_consequence = info["moral_consequence"]
    immoral_action = info["immoral_action"]
    immoral_consequence = info["immoral_consequence"]

    instruction = "You are a moral logic reasoner. Your first task is to generate a reverse norm based on a moral action for a given situation and intention. For example, Situation: Josh drives Penelope to work every day and is hanging outside her house to pick her up, but there is no sign of her. Actor's Intention: Josh needs to get to work. Action: Josh continues to sit outside Penelope's house until she walks out the door a few minutes later. Norm: It is unwise to prioritize anything over your job. Followed the given example, write a reverse norm for the following scenario to make the action meaningless. Situation: {}. Actor's intention: {}. Action: {}. The norm should be short and easy to understand like the norm in the example".format(situation, intention, moral_action)

    instruction2 = "Your second task is to generate an vague norm based on an immoral action for a given situation and intention. For example, Situation: Josh drives Penelope to work every day and is hanging outside her house to pick her up, but there is no sign of her. Actor's Intention: Josh needs to get to work. Action: Josh shrugs his shoulders and drives away to work after sitting around a moment. Norm: Getting to work in time is important for one's career. Followed the given example, write an ambiguous norm for the following scenario to make the action meaningful. Situation: {}. Action's intention: {}. Action: {}. The norm should be short and easy to understand like the norm in the example".format(situation, intention, immoral_action)

    instruction3 = "Your third task is to analyze the impact of a norm related to the situation. Analyze the following aspects: \n 1. Core Scope of Authority: 1) Personal or internal; 2) Interpersonal or relational; 3) Communal or Organizational; 4) Societal or Legal; 5) Universal or Humanistic. \n 2. Cultural Universality: highly universal, or universal with variations, or culture-specific, or highly contested/subcultural. \n 3. Contextual Dependency: high generalizable, or moderately Dependent, or highly dependent. \n Analyze Norm: {} Generate the choices between the candidates with explanation. Return all the results in JSON format with id attribute as {}.".format(norm, ID)

    print(j, ID)
    print("========")
    #print(instruction)
    #print(instruction2)
    #print("\n")


    for i in range(5):
        try:
            print("attemp: {}".format(i))
            response = client.models.generate_content(
                model="gemini-2.5-pro", contents=instruction+"\n"+instruction2+"\n"+instruction3,
                config={'response_mime_type': 'application/json'},
            )
    
            print(response.text)
            fout = open(os.path.join(save_dir, str(j)+".json"), "w")
            json.dump(json.loads(response.text), fout, indent=4)
            fout.close()
            time.sleep(1)
            break
        except:
            print("no response")
    #break
