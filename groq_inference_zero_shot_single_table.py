import os
import json
import pandas as pd
from groq import Groq
import time
import random

client = Groq(
    api_key="",
    )
table_selection = "last"
def create_prompt(context,question):
    return f"""<s>[INST]Below is a table provided for an entity. Utilizing this table, answer the following question. The answer should be concise, i.e., within 5 to 10 words.
table: {context}
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
    if entity != "England_cricket_team":
        continue
    folder_question = 0
    entity_name = entity.split(" ")
    entity_name = "_".join(entity_name)
    with open(f'data/cricket_team/{entity_name}.json','r') as f:
        timeline = json.load(f)
    if len(timeline)>20:
        continue
    # timeline = json.dumps(timeline)
    table = None
    if table_selection == "first":
        first_key = next(iter(timeline))
        table = timeline[first_key]
        table["date_time"] = first_key
    elif table_selection == "last":
        last_key = list(timeline.keys())[-1]
        table = timeline[last_key]
        table["date_time"] = last_key
    elif table_selection == "random":
        no_of_tables = len(timeline)
        table_number = random.randint(1,no_of_tables-2)
        timeline_keys = list(timeline.keys())
        table = timeline[timeline_keys[table_number]]
        table["date_time"] = timeline_keys[table_number]
    # timeline = json.dumps(timeline)
    df = {"entity": [],
          "question": [],
          "actual_answer": [],
          "predicted_answer": []}
    for questions_category in entities[entity]:
        for question in entities[entity][questions_category]:
            actual_question = entities[entity][questions_category][question]["Q"]
            prompt = create_prompt(table,actual_question)
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
    dataframe.to_csv(f"data/cricket_team/predictions/single-stage/zero_shot_single_table/last_table/llama-3-70b/{entity}.csv", index=False)
print(f"total - {total_questions}")