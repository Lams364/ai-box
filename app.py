import token
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

app = Flask(__name__)

# Get the Hugging Face token from environment variables
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# Add the token to the API call when loading the model
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Change to any model you'd like
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HUGGING_FACE_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HUGGING_FACE_TOKEN)


@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json.get("prompt", "")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
