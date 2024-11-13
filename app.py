import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from models import ChangeModelRequest, ChangeTokenRequest, PredictRequest
from utils import ModelManager, build_prompt, generate_model_response

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="ai-box API Server",
    description="API interface that covers LLM generation with huggingface models",
    version="0.1.0",
)

MODEL_DIR = os.getenv("MODEL_DIR")
if MODEL_DIR in [None, ""]:
    MODEL_DIR = "./hf_local_models"
os.makedirs(MODEL_DIR, exist_ok=True)

model_manager = ModelManager(MODEL_DIR)


def get_model_manager():
    return model_manager


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
    model = model_manager.model
    tokenizer = model_manager.tokenizer
    prompt = request.prompt
    max_tokens = request.max_new_tokens

    content = generate_model_response(model, tokenizer, prompt, max_tokens)

    logger.info(f"Generated response for prompt: {prompt}")
    return {"content": content}


@app.post(
    "/generate/log-advice",
    summary="Generate response from model to help with logging",
    response_description="Model response",
)
async def predict(request: PredictRequest):
    """
    Generate a response from the model based on the provided code.

    - **prompt**: The code snippet for the model to generate a response.
    - **max_new_tokens**: The maximum number of new tokens to generate (default is 128).

    Returns:
    - **content**: The generated response from the model.
    """
    model = model_manager.model
    tokenizer = model_manager.tokenizer
    prompt = request.prompt
    max_tokens = request.max_new_tokens

    context = f"""
    You are an AI assistant helping a developer.\n
    Your task is to add simples logging steps to the code.\n
    Do not change the code, or add methods.\n
    Here is the code :\n
    """

    contexted_prompt = build_prompt(context, prompt)

    content = generate_model_response(model, tokenizer, contexted_prompt, max_tokens)
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
    completed = model_manager.set_model(hf_model_id)
    return {"completed": completed, "model_name": model_manager.model_name}


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
    completed = model_manager.set_token(hf_token)
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
    - **model_dir**: The directory where the model is stored.
    """
    return {
        "model_name": model_manager.model_name or "No model set",
        "device": model_manager.device or "Unknown",
        "model_dir": model_manager.model_dir,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
