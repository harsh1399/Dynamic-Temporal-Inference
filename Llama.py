from openai import OpenAI
from colorama import init
from colorama import Fore, Back, Style
import time
from tokenizers import Tokenizer
import json
import pandas as pd


init()



time.sleep(5)

# prompts = [
#     "what is ROI in the context of finance, provide a worked example?",
#     "define the efficient frontier in the context of finance",
#     "what is glass stegal?",
#     "how does derivative pricing work?",
# ]

def create_prompt(timeline,question):
    return f"""
<s>[INST]Below is a set of tables provided for an entity across some timeline that captures the information evolving for that entity across time. Utilizing this entity timeline, answer the following question. Also, the answers should be concise, i.e., within 10 to 15 words. Further, answer the question based solely on the information presented in the timeline without referencing any external data or information.
timeline: {timeline}
question: {question}
answer:[/INST]
"""

with open('data/cricket_team/cricket_questions.json','r') as f:
    entities = json.load(f)

df = {"entity":[],
      "question":[],
      "actual_answer":[],
      "predicted_answer":[]}

for entity in entities:
    entity_name = entity.split(" ")
    entity_name = "_".join(entity_name)
    with open(f'data/cricket_team/{entity_name}.json','r') as f:
        timeline = json.load(f)
    if len(timeline)>20:
        continue
    print(entity)
    timeline = json.dumps(timeline)
    for questions_category in entities[entity]:
        for question in entities[entity][questions_category]:
            prompt = create_prompt(timeline,entities[entity][questions_category][question]["Q"])
            # print(f"Prompt:{prompt}")
            client = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="123",
            )
            temp_prompt = "<|begin_of_text|>Hi how are you?"
            # model="mistralai/Mistral-7B-v0.1",
            response = client.chat.completions.create(
                model = "meta-llama/Meta-Llama-3-8B",
                messages=[
                    {"role": "user","content": temp_prompt}
                ]
            )
            print(f"Response: {response}")
            predicted_ans = response.choices[0].message.content
            print(f"predicted ans: {predicted_ans}")
            # actual_ans = entities[entity][questions_category][question]["A"]
            # df["entity"].append(entity)
            # df["question"].append(entities[entity][questions_category][question]["Q"])
            # df["actual_answer"].append(actual_ans)
            # df["predicted_answer"].append(predicted_ans)
            break
        break
    break
# dataframe = pd.DataFrame(df)
# dataframe.to_csv("data/cricket_team/llama_results.csv",index=False)