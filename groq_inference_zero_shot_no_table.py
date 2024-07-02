import os
import json
import pandas as pd
from groq import Groq
import time
import random

client = Groq(
    api_key="",
    )
def create_prompt(question):
    return f"""<s>[INST]Answer the following question concisely, i.e., within 5 to 10 words.
question: {question}
answer:[/INST]
"""

def generate_response(model,prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response

with open('data/cricket_team/cricket_questions_latest.json','r') as f:
    entities = json.load(f)
total_questions = 0

for entity in entities:
    if entity == "Bermuda_national_cricket_team" or entity == "England_cricket_team" or entity == "West_Indies_cricket_team":
        folder_question = 0
        df = {"entity": [],
              "question": [],
              "actual_answer": [],
              "predicted_answer": []}
        for questions_category in entities[entity]:
            for question in entities[entity][questions_category]:
                actual_question = entities[entity][questions_category][question]["Q"]
                prompt = create_prompt(actual_question)
                try:
                    response = generate_response("llama3-70b-8192",prompt)
                except:
                    time.sleep(30)
                    response = generate_response("llama3-70b-8192",prompt)
                predicted_ans = ""
                try:
                    predicted_ans = response.choices[0].message.content
                except:
                    try:
                        time.sleep(70)
                        response = generate_response("llama3-70b-8192", prompt)
                    except:
                        print(entity, questions_category, actual_question)
                        continue
                actual_ans = entities[entity][questions_category][question]["A"]
                df["entity"].append(entity)
                df["question"].append(entities[entity][questions_category][question]["Q"])
                df["actual_answer"].append(actual_ans)
                df["predicted_answer"].append(predicted_ans)
                time.sleep(5)
                folder_question += 1
        total_questions += folder_question
        print(f"{entity} -- {folder_question}")
        dataframe = pd.DataFrame(df)
        dataframe.to_csv(f"data/cricket_team/predictions/single-stage/zero_shot_no_table/llama-3-70b/{entity}.csv", index=False)
print(f"total - {total_questions}")