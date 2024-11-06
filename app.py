from dotenv import load_dotenv
import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from utils import ModelManager

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

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
os.makedirs(MODEL_DIR, exist_ok=True)

model_manager = ModelManager(MODEL_DIR)


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


model_manager.init_model()


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

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(
        model_manager.device
    )
    inputs["attention_mask"] = (
        inputs["attention_mask"] if "attention_mask" in inputs else None
    )
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    content = tokenizer.decode(outputs[1], skip_special_tokens=True)
    # paramètres pour generate pour retourner réponse uniquement
    # TAG pour modèles importants
    # Expressions régulière pour r
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
    completed = model_manager.set_model_name(hf_model_id)
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
    """
    return {"model_name": model_manager.model_name, "device": str(model_manager.device)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
