# app.py
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
import torch
import os

app = Flask(__name__)

MODEL_DIR = './models'

os.makedirs(MODEL_DIR, exist_ok=True)

# Define globals
MODEL = None
TOKEN = None
TOKENIZER = None
DEVICE = None
MODEL_NAME = None

def validate_huggingface_token(token: str) -> bool:
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
    return response.status_code == 200

def set_model_name(model_name: str):
    global MODEL, TOKEN, TOKENIZER, DEVICE, MODEL_NAME  # Reference the globals
    response = create_model(model_name, TOKEN)
    if response is not None:
        MODEL, TOKENIZER, DEVICE = response
        MODEL_NAME = model_name
        return True
    else:
        return False
    
def set_token(new_token:str):
    global MODEL, TOKEN, TOKENIZER, DEVICE, MODEL_NAME  # Reference the globals   
    validate = validate_huggingface_token(new_token)
    response = create_model(MODEL_NAME, new_token)
    if validate and response is not None:
        MODEL, TOKENIZER, DEVICE = response
        TOKEN = new_token
        return True
    else:
        return False


def create_model(model_name: str, token: str):
    try:
        # Check if the model is already in the model directory, otherwise download it
        if not os.path.exists(os.path.join(MODEL_DIR, model_name)):
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=MODEL_DIR, token=token)
            # Add a new [PAD] token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})    
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, cache_dir=MODEL_DIR, token=token, pad_token_id=tokenizer.eos_token_id)
        else:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, token=token)
            # Add a new [PAD] token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True, token=token, pad_token_id=tokenizer.eos_token)
        
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        return model, tokenizer, device
    except Exception as e:
        return None
    
def init_model():
    global MODEL, TOKEN, MODEL_NAME  # Reference the globals
    # Model details
    # meta-llama/Llama-3.2-1B-Instruct
    # Qwen/Qwen2.5-Coder-1.5B-Instruct

    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    TOKEN = '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'
    set_model_name(MODEL_NAME)
    print("Device : " + str(DEVICE))     

init_model()

@app.route('/predict', methods=['POST'])
def predict():
    """
    POST endpoint to get a response for the model.
    
    This endpoint allows the client to ask a question to the model. It accepts a JSON payload with a `prompt` key for the user input query.
    The `max_new_tokens` key specifies the maximum number of new tokens the model should generate in response to the input prompt, 
    if not specified, 128 by default.

    Usage:
        [POST] /predict
        JSON body: { "prompt": "USER_TEXT_PROMPT", "max_new_tokens": MAX_TOKENS }

    Returns:
        JSON response with:
            - `content` (str): Model response.
    """
    data = request.json
    if 'prompt' not in data:
        return jsonify({'error': 'No prompt provided'}), 400
    
    model = MODEL
    tokenizer = TOKENIZER
    prompt = data['prompt']
    max_tokens = data.get('max_new_tokens', 128)
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    inputs['attention_mask'] = inputs['attention_mask'] if 'attention_mask' in inputs else None
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    content = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({'content': content})

@app.route('/change_model', methods=['POST'])
def change_model():
    """
    POST endpoint to change the active model in the application.
    
    This endpoint allows the client to update the current model by specifying
    the model ID. It accepts a JSON payload with a `model_id` key and 
    responds with the operation status.

    Usage:
        [POST] /change_model
        JSON body: { "model_id": "MODEL_ID" }

    Returns:
        JSON response with:
            - `completed` (bool): True if the model change was successful, False otherwise.
            - `model_name` (str): The name of the model currently in use (for confirmation).
    """
    if 'model_id' not in request.json:
        return jsonify({'error': 'No model_id provided'}), 400
    
    model_id = request.json.get('model_id')
    completed = set_model_name(model_id)
    return jsonify({'completed': completed, "model_name": MODEL_NAME})

@app.route('/change_token', methods=['POST'])
def change_token():
    """
    POST endpoint to change the active huggingface token in the application.
    
    This endpoint allows the client to update the current huggingface token by specifying
    a token. It accepts a JSON payload with a `token` key and 
    responds with the operation status.

    Usage:
        [POST] /change_model
        JSON body: { "token": "HUGGINGFACE_TOKEN" }

    Returns:
        JSON response with:
            - `completed` (bool): True if the token change was successful, False otherwise.
    """
    if 'token' not in request.json:
        return jsonify({'error': 'No token provided'}), 400
    token = request.json.get('token')
    completed = set_token(token)
    return jsonify({'completed': completed})

@app.route('/model_info')
def model_info():
    """
    GET endpoint to get information about the active model in the application.
    
    This endpoint allows the client to get basic information about the active model in the application.

    Usage:
        [GET] /model_info

    Returns:
        JSON response with:
            - `model_name` (str): The name of the model currently in use.
            - `device` (str): Information about the device executing the model (cpu) or (cuda)
    """
    return jsonify({'model_name': MODEL_NAME, "device": str(DEVICE)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)

