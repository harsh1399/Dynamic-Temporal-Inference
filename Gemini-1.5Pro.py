import vertexai
from vertexai.preview.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
import pandas as pd
import time
import json

project_id = "dynamic-temporal"
vertexai.init(project=project_id, location="us-central1")

def create_prompt(timeline,question):
    return f"""
Below is a set of tables provided for an entity across some timeline that captures the information evolving for that entity across time. Utilizing this entity timeline, answer the following question. Also, the answers should be concise, i.e., within 5 to 10 words. Further, answer the question based solely on the information presented in the timeline without referencing any external data or information.
timeline: {timeline}
question: {question}
answer:
"""

safety_config = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

# answer = gemini_pro_model.generate_content(safety_settings=safety_config)
def generate_response(model,prompt):
    model = GenerativeModel(model_name=model)
    response = model.generate_content(prompt,safety_settings=safety_config)
    return response

with open('data/cricket_team/cricket_questions_gemini.json','r') as f:
    entities = json.load(f)
total_questions = 0
for entity in entities:
    folder_question = 0
    entity_name = entity.split(" ")
    entity_name = "_".join(entity_name)
    with open(f'data/cricket_team/{entity_name}.json','r') as f:
        timeline = json.load(f)
    if len(timeline)>20:
        continue
    timeline = json.dumps(timeline)
    df = {"entity": [],
          "question": [],
          "actual_answer": [],
          "predicted_answer": []}
    for questions_category in entities[entity]:
        for question in entities[entity][questions_category]:
            actual_question = entities[entity][questions_category][question]["Q"]
            prompt = create_prompt(timeline,actual_question)
            try:
                response = generate_response("gemini-1.5-pro-001",prompt)
            except Exception as ex:
                print(ex)
                time.sleep(90)
                try:
                    response = generate_response("gemini-1.5-pro-001",prompt)
                except:
                    print(entity,questions_category)
                    break
            predicted_ans = response.text
            actual_ans = entities[entity][questions_category][question]["A"]
            df["entity"].append(entity)
            df["question"].append(entities[entity][questions_category][question]["Q"])
            df["actual_answer"].append(actual_ans)
            df["predicted_answer"].append(predicted_ans)
            time.sleep(30)
            folder_question += 1
    total_questions += folder_question
    print(f"{entity} -- {folder_question}")
    dataframe = pd.DataFrame(df)
    dataframe.to_csv(f"data/cricket_team/new_predictions/gemini-1.5-pro/{entity}.csv", index=False)
print(f"total - {total_questions}")