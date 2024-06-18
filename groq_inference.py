import os
import json
import pandas as pd
from groq import Groq
import time

client = Groq(
    api_key="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",
    )

def create_prompt(timeline,question):
    return f"""
<s>[INST]Below is a set of tables provided for an entity across some timeline that captures the information evolving for that entity across time. Utilizing this entity timeline, answer the following question. Also, the answers should be concise, i.e., within 5 to 10 words. Further, answer the question based solely on the information presented in the timeline without referencing any external data or information.
timeline: {timeline}
question: {question}
answer:[/INST]
"""

with open('data/cricket_team/cricket_questions_selected.json','r') as f:
    entities = json.load(f)

for entity in entities:
    entity_name = entity.split(" ")
    entity_name = "_".join(entity_name)
    with open(f'data/cricket_team/{entity_name}.json','r') as f:
        timeline = json.load(f)
    if len(timeline)>20:
        continue
    print(entity)
    timeline = json.dumps(timeline)
    df = {"entity": [],
          "question": [],
          "actual_answer": [],
          "predicted_answer": []}
    for questions_category in entities[entity]:
        for question in entities[entity][questions_category]:
            actual_question = entities[entity][questions_category][question]["Q"]
            if questions_category == "q15":
                actual_question += "(Answer in decimal form.)"
            prompt = create_prompt(timeline,actual_question)
            response = client.chat.completions.create(
                model = "llama3-70b-8192",
                messages=[
                    {"role": "user","content": prompt}
                ]
            )
            predicted_ans = response.choices[0].message.content
            print(f"question: {actual_question}, predicted ans: {predicted_ans}")
            actual_ans = entities[entity][questions_category][question]["A"]
            df["entity"].append(entity)
            df["question"].append(entities[entity][questions_category][question]["Q"])
            df["actual_answer"].append(actual_ans)
            df["predicted_answer"].append(predicted_ans)
            time.sleep(60)
    dataframe = pd.DataFrame(df)
    dataframe.to_csv(f"data/cricket_team/new_predictions/{entity}.csv", index=False)
