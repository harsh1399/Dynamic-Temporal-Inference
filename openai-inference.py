import vertexai
import openai
import json
import time
import pandas as pd
from google.auth import default
from google.auth.transport.requests import Request

# TODO(developer): Update and un-comment below lines
project_id = "dynamic-temporal"
location = "us-central1"
vertexai.init(project=project_id, location=location)

def create_prompt(timeline,question):
    return f"""
Below is a set of tables provided for an entity across some timeline that captures the information evolving for that entity across time. Utilizing this entity timeline, answer the following question. Also, the answers should be concise, i.e., within 5 to 10 words. Further, answer the question based solely on the information presented in the timeline without referencing any external data or information.
timeline: {timeline}
question: {question}
answer:
"""

def generate_response(model,prompt):
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    auth_request = Request()
    credentials.refresh(auth_request)
    # # OpenAI Client
    client = openai.OpenAI(
        base_url=f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/endpoints/openapi",
        api_key=credentials.token,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response

with open('data/economy/economy_questions.json','r') as f:
    entities = json.load(f)
total_questions = 0
for entity in entities:
    folder_question = 0
    entity_name = entity.split(" ")
    entity_name = "_".join(entity_name)
    with open(f'data/economy/{entity_name}.json','r') as f:
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
                response = generate_response("google/gemini-1.5-pro-001",prompt)
            except:
                time.sleep(90)
                response = generate_response("google/gemini-1.5-pro-001",prompt)
            # print(response)
            predicted_ans = ""
            try:
                predicted_ans = response.choices[0].message.content
            except:
                try:
                    time.sleep(120)
                    response = generate_response("google/gemini-1.5-pro-001",prompt)
                except:
                    print(entity,questions_category,actual_question)
                    continue
            actual_ans = entities[entity][questions_category][question]["A"]
            df["entity"].append(entity)
            df["question"].append(entities[entity][questions_category][question]["Q"])
            df["actual_answer"].append(actual_ans)
            df["predicted_answer"].append(predicted_ans)
            time.sleep(15)
            folder_question += 1
    total_questions += folder_question
    print(f"{entity} -- {folder_question}")
    dataframe = pd.DataFrame(df)
    dataframe.to_csv(f"data/economy/predictions/gemini-1.5-pro/{entity}.csv", index=False)
print(f"total - {total_questions}")