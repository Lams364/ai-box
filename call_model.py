import re
import requests


def call_model(data):
    url = "http://localhost:8080/completion"
    response = requests.post(url, json=data)
    return response.json()


def prepare_prompt(prompt):
    prompt = (
        "Context: You are an AI assistant that helps people with their questions. "
        + "Answer only the question you are being asked. Don't add questions that is not in the prompt. Be consise. "
        + "Don't add an introduction or any form of 'A:' to your answer. Just answer the question after the 'QUESTION:' tag. "
        + "QUESTION:\n\n"
        + prompt
    )
    return {"prompt": prompt, "max_tokens": 100, "temperature": 0.1}


data = prepare_prompt("What's a docker environment?")
response = call_model(data)
print(f"Prompt:\n{data['prompt']}\n\nResponse:\n{response['content']}")
