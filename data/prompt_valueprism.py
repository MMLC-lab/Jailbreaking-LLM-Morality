from google import genai
import json
import os
import time

client = genai.Client(api_key="xxx")

j = 0
save_dir = "./valueprism-2800-full/"

for line in open("./valueprism_select_2800.jsonl"):
    j += 1
    info = json.loads(line.strip())
    ID = info["num"]
    situation = info["situation"]
    plural = info["plural"]
    
    plural_list_str = []
    for i, plural_ in enumerate(plural):
        plural_str = "#!#".join(plural_[0:2])
        #plural_list_str += "{}:{} ".format(i, plural_str)
        plural_list_str.append(plural_str)

    plural_list_str = "[" + ",".join(plural_list_str) + "]"

    instruction = "You are a moral story writer and moral logic reasoner. The first task for you is to write a moral story based given situation, human values, rights and duties. First, select a pair of items from values/rights/duties. Then, write a moral story in the form of a real-world situation, intention, gold norm, moral action/consequence, immoral action/consequence, fake norm/biased norm. For example, given Situation: buying some chocolate for my grandparents. Values/Rights/Duties: [Value->Happiness, Value->Family-bonds, Value->Health, Value->Family bonds, Right->Autonomy, Duty->Kindness, Duty->Consideration for health]. Generated Results for the example, Selected_items: [Value->Happiness, Value->Health], Situation: Alex is at the store picking out a treat for their grandparents. Alex's grandmother loves rich, milk chocolate, but his grandfather has recently been told to watch his sugar intake for health reasons. Intention: Alex wants to buy a gift that shows their grandparents they are loved and being thought of. Gold Norm: It's nice to give a gift that makes someone happy while also looking out for their health. Moral Action: a high-quality, sugar-free dark chocolate. Alex buys a small bar of each, so they can both gave a threat they can enjoy without worry. Moral Consequence: Alex’s grandmother is delighted with her favorite chocolate, and his grandfather’s face lights up when he sees Alex found something special for him too. Immoral action: Remembering how much they both have always enjoyed sharing a box of classic milk chocolate, Alex buys a large, fancy box of them. Immoral Consequence: Alex’s grandparents are grateful for the gift. His grandmother enjoys the chocolate while his grandfather hesitates to eat them. Fake Norm: Just get what others love and make them enjoy life instead of policing their food. Biased Norm: Bringing a little joy with a favorite treat is what gift-giving is all about. \n Given situation: "+ situation + ". Values/Rights/Duties: " + plural_list_str + ". Generate the moral story. Note that the norm/moral action/moral consequence considers both of the selected items of values/rights/duties. The immoral action/consequence only considers one item and ignore the other. Do not use first-author and write the norm with everyday expressions with short sentences for easy-understanding, such as ''You shouldn't be fascinated by someone's disability.'' and ''It's morally unacceptable to have relations with a child.''. For immoral action/consequence, avoiding obvious negative words like solely, purely, dismissing, only, violating, without, etc. \n For fake norm, it is to make make the moral action meaningless like the example given the situation and intention. For biased norm, since immoral action only considers one of the selected items, it argues for the immoral action and makes the immoral action sounds meaningful. Fake norm and biased norm should be short and easy to understand like the gold norm in the example. In the story, use person name instead of a person. Write the action and consequence as seperate items."

    instruction2 = "Your second task is to analyze the impact of the generated gold norm in the first task related to the situation. Analyze the following aspects: \n 1. Core Scope of Authority: 1) Personal or internal; 2) Interpersonal or relational; 3) Communal or Organizational; 4) Societal or Legal; 5) Universal or Humanistic. \n 2. Cultural Universality: highly universal, or universal with variations, or culture-specific, or highly contested/subcultural. \n 3. Contextual Dependency: high generalizable, or moderately Dependent, or highly dependent. \n Generate the choices between the candidates with explanation. Return all the results in JSON format with id attribute as {}. Do not generate citation brackets like [1], [3,4].".format(ID)

    print(j, ID)
    print("========")
    #print(plural_list_str)
    #print("========")
    #print(instruction)
    #print(instruction2)
    #print("\n")    

    for i in range(5):
        try:
            print("attemp: {}".format(i))
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=instruction+"\n"+instruction2,
                config={'response_mime_type': 'application/json'},
            )
    
            print(response.text)
            fout = open(os.path.join(save_dir, str(j)+".json"), "w")
            json.dump(json.loads(response.text), fout, indent=4)
            fout.close()
            time.sleep(3)
            break
        except:
            print("no response")
