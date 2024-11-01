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

    # TOKEN1 = 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
    # TOKEN2 = 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
    set_model_name(MODEL_NAME)
    print(next(MODEL.parameters()).device)

def test_token(token:str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=MODEL_DIR, token=TOKEN)       

init_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    model = MODEL
    tokenizer = TOKENIZER
    text = data['text']
    nb_tokens = data.get('tokens', 128)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    inputs['attention_mask'] = inputs['attention_mask'] if 'attention_mask' in inputs else None
    outputs = model.generate(**inputs, max_new_tokens=nb_tokens)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({'content': result})

@app.route('/change_model')
def change_model():
    # Use like this /change_model?name=MODEL_NAME(HUGGINGFACE)
    name = request.args.get('name')
    completed = set_model_name(name)
    return jsonify({'completed': completed, "model_name": MODEL_NAME})

@app.route('/change_token')
def change_token():
    # Use like this /change_token?token=HUGGINGFACE_TOKEN
    token = request.args.get('token')
    completed = set_token(token)
    return jsonify({'completed': completed})

@app.route('/model_info')
def model_info():
    return jsonify({'model': MODEL_NAME, "device": str(DEVICE)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)

