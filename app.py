import logging
import os

import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, field_validator, validator
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ai-box API Server",
    description="API interface that covers LLM generation with huggingface models",
    version="0.1.0",
)

MODEL_DIR = "./models"

os.makedirs(MODEL_DIR, exist_ok=True)

# Define globals
MODEL = None
TOKEN = None
TOKENIZER = None
DEVICE = None
MODEL_NAME = None


class PredictRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128

    @field_validator("prompt")
    def validate_prompt(cls, prompt):
        if not prompt:
            raise ValueError("No prompt provided")
        if len(prompt) > 2048:
            raise ValueError("Prompt is too long")
        return prompt


class ChangeModelRequest(BaseModel):
    hf_model_id: str

    @field_validator("hf_model_id")
    def validate_hf_model_id(cls, hf_model_id):
        if not hf_model_id:
            raise ValueError("No hf_model_id provided")
        return hf_model_id


class ChangeTokenRequest(BaseModel):
    hf_token: str

    @field_validator("hf_token")
    def validate_token(cls, hf_token):
        if not hf_token:
            raise ValueError("No hf_token provided")
        return hf_token


def validate_huggingface_token(hf_token: str) -> bool:
    headers = {"Authorization": f"Bearer {hf_token}"}
    response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
    return response.status_code == 200


def set_model_name(model_name: str):
    global MODEL, TOKEN, TOKENIZER, DEVICE, MODEL_NAME  # Reference the globals
    response = create_model(model_name, TOKEN)
    if response is not None:
        MODEL, TOKENIZER, DEVICE = response
        MODEL_NAME = model_name
        logger.info(f"Model set to {model_name}")
        return True
    else:
        logger.error(f"Failed to set model to {model_name}")
        return False


def set_token(new_token: str):
    global MODEL, TOKEN, TOKENIZER, DEVICE, MODEL_NAME  # Reference the globals
    validate = validate_huggingface_token(new_token)
    response = create_model(MODEL_NAME, new_token)
    if validate and response is not None:
        MODEL, TOKENIZER, DEVICE = response
        TOKEN = new_token
        logger.info("Token updated successfully")
        return True
    else:
        logger.error("Failed to update hf_token")
        return False


def create_model(model_name: str, hf_token: str):
    try:
        # Check if the model is already in the model directory, otherwise download it
        if not os.path.exists(os.path.join(MODEL_DIR, model_name)):
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=MODEL_DIR,
                token=hf_token,
            )
            # Add a new [PAD] hf_token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=MODEL_DIR,
                token=hf_token,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_DIR, trust_remote_code=True, hf_token=hf_token
            )
            # Add a new [PAD] hf_token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR,
                trust_remote_code=True,
                token=hf_token,
                pad_token_id=tokenizer.eos_token,
            )

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(f"Model {model_name} loaded successfully on {device}")
        return model, tokenizer, device
    except Exception as e:
        logger.error(f"Failed to create model {model_name}: {e}")
        return None


def init_model():
    global MODEL, TOKEN, MODEL_NAME  # Reference the globals
    # Model details
    # meta-llama/Llama-3.2-1B-Instruct
    # Qwen/Qwen2.5-Coder-1.5B-Instruct

    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    TOKEN = "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"
    set_model_name(MODEL_NAME)
    logger.info(f"Device: {DEVICE}")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    simplified_errors = []
    for error in exc.errors():
        field = error["loc"][-1]  # Field name (e.g., "prompt")
        message = error["msg"]  # Error message (e.g., "Field required")
        simplified_errors.append(f"{field}: {message}")

    return JSONResponse(
        status_code=400,
        content={"errors": simplified_errors},
    )


init_model()


@app.post(
    "/predict",
    summary="Generate response from model",
    response_description="Model response",
)
async def predict(request: PredictRequest):
    """
    Generate a response from the model based on the provided prompt.

    - **prompt**: The input text for the model to generate a response.
    - **max_new_tokens**: The maximum number of new tokens to generate (default is 128).

    Returns:
    - **content**: The generated response from the model.
    """
    model = MODEL
    tokenizer = TOKENIZER
    prompt = request.prompt
    max_tokens = request.max_new_tokens

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(
        DEVICE
    )
    inputs["attention_mask"] = (
        inputs["attention_mask"] if "attention_mask" in inputs else None
    )
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    content = tokenizer.decode(outputs[0], skip_special_tokens=True)

    logger.info(f"Generated response for prompt: {prompt}")
    return {"content": content}


@app.post(
    "/change_model",
    summary="Change the active model",
    response_description="Operation status",
)
async def change_model(request: ChangeModelRequest):
    """
    Change the active model in the application.

    - **hf_model_id**: The Hugging Face model ID to switch to.

    Returns:
    - **completed**: True if the model change was successful, False otherwise.
    - **model_name**: The name of the model currently in use.
    """
    if not request.hf_model_id:
        logger.error("No hf_model_id provided")
        raise HTTPException(status_code=400, detail="No hf_model_id provided")

    hf_model_id = request.hf_model_id
    completed = set_model_name(hf_model_id)
    return {"completed": completed, "model_name": MODEL_NAME}


@app.post(
    "/change_token",
    summary="Change the Hugging Face token",
    response_description="Operation status",
)
async def change_token(request: ChangeTokenRequest):
    """
    Change the active Hugging Face token in the application.

    - **hf_token**: The new Hugging Face token.

    Returns:
    - **completed**: True if the token change was successful, False otherwise.
    """
    if not request.hf_token:
        logger.error("No hf_token provided")
        raise HTTPException(status_code=400, detail="No hf_token provided")
    hf_token = request.hf_token
    completed = set_token(hf_token)
    return {"completed": completed}


@app.get(
    "/model_info",
    summary="Get active model information",
    response_description="Model information",
)
async def model_info():
    """
    Get information about the active model in the application.

    Returns:
    - **model_name**: The name of the model currently in use.
    - **device**: Information about the device executing the model (cpu or cuda).
    """
    return {"model_name": MODEL_NAME, "device": str(DEVICE)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
