import os
import json
import pandas as pd
from groq import Groq
import time

client = Groq(
    api_key="",
    )

def create_prompt(few_shot_timeline,few_shot_question,few_shot_answer,timeline,question):
    return f"""<s>[INST]Below is a set of tables provided for an entity across some timeline that captures the information evolving for that entity across time. Utilizing this entity timeline, answer the following question. Also, the answer should be concise, i.e., within 5 to 10 words. Further, answer the question based solely on the information presented in the timeline without referencing any external data or information.
Here's an example for reference:
Example - 
Timeline: {few_shot_timeline}
question: {few_shot_question}
answer: {few_shot_answer}
Now, answer the following question using the below given timeline. 
timeline: {timeline}
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
with open('data/cricket_team/few_shot_example.json','r') as f:
    few_shot_timeline = json.load(f)
with open('data/cricket_team/few_shot_questions.json','r') as f:
    few_shot_questions = json.load(f)
few_shot_keys = few_shot_questions.keys()
total_questions = 0
for entity in entities:
    if entity != "India_national_cricket_team":
        continue
    folder_question = 0
    entity_name = entity.split(" ")
    entity_name = "_".join(entity_name)
    with open(f'data/cricket_team/{entity_name}.json','r') as f:
        timeline = json.load(f)
    if len(timeline)>20:
        continue
    timeline = json.dumps(timeline)
    few_shot_timeline = json.dumps(few_shot_timeline)
    df = {"entity": [],
          "question": [],
          "actual_answer": [],
          "predicted_answer": []}
    for questions_category in entities[entity]:
        for question in entities[entity][questions_category]:
            if questions_category not in few_shot_keys:
                continue
            few_shot_example_question = few_shot_questions[questions_category]
            actual_question = entities[entity][questions_category][question]["Q"]
            prompt = create_prompt(few_shot_timeline,few_shot_example_question['Q'],few_shot_example_question['A'],timeline,actual_question)
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
    dataframe.to_csv(f"data/cricket_team/predictions/single-stage/few_shot_multitable/llama-3-70b/{entity}.csv", index=False)
print(f"total - {total_questions}")