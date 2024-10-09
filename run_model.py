# app.py
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = Flask(__name__)

# Define the model directory (persistent volume)
MODEL_DIR = './models'

# Ensure the directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Model details
model_name = "meta-llama/Llama-3.2-1B-Instruct"
token = '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'

# Check if the model is already in the model directory, otherwise download it
if not os.path.exists(os.path.join(MODEL_DIR, model_name)):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_DIR, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=MODEL_DIR, token=token)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, token=token)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, token=token)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    inputs = tokenizer(data['text'], return_tensors="pt").to(device)
    outputs = model.generate(inputs.input_ids)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
