import logging
import os

import requests
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

delimiter = "\n\n~~~~~"


def validate_huggingface_token(hf_token: str) -> bool:
    headers = {"Authorization": f"Bearer {hf_token}"}
    response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
    return response.status_code == 200


def generate_model_response(model, tokenizer, prompt, max_tokens, temperature=0.2):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs["attention_mask"] = (
        inputs["attention_mask"] if "attention_mask" in inputs else None
    )

    device = next(model.parameters()).device  # Automatically detects the model’s device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        repetition_penalty=1.2,
        do_sample=True,
    )
    content = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    content = content.split(delimiter)[-1].strip()
    return content


def build_prompt(context, prompt):
    return ("[INST]" + context + prompt + "[/INST]" + delimiter).strip()


class ModelManager:

    token: str
    model_dir: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device
    model_name: str

    def __init__(self, model_dir: str):
        self.model_dir = model_dir

        # Model details
        # meta-llama/Llama-3.2-1B-Instruct
        # Qwen/Qwen2.5-Coder-1.5B-Instruct
        # Qwen/Qwen2.5-0.5B

        model_name = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
        token = os.getenv("HF_TOKEN")
        self.set_token(token)
        self.set_model(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"Memory allocated on GPU: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
        )

    def create_model(self, model_name: str):
        try:
            quant_config = (
                BitsAndBytesConfig(load_in_8bit=True)
                if torch.cuda.is_available()
                else None
            )

            if not self.token:
                logger.warning(
                    "Hugging Face token is not provided. Initializing without token."
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=self.model_dir,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=self.model_dir,
                    pad_token_id=tokenizer.eos_token_id,
                    quantization_config=quant_config,
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=self.model_dir,
                    token=self.token,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=self.model_dir,
                    token=self.token,
                    pad_token_id=tokenizer.eos_token_id,
                    quantization_config=quant_config,
                )

            # Add a new [PAD] token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            logger.info(f"Model {model_name} created successfully")

            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {e}")
            return None

    def set_model(self, model_name: str):
        logger.info(f"Setting model to {model_name}")
        response = self.create_model(model_name)
        if response is not None:
            self.model, self.tokenizer = response
            self.model_name = model_name
            logger.info(f"Model successfully set to {model_name}")
            return True
        else:
            logger.error(f"Failed to set model to {model_name}")
            self.model_name = None  # Fallback to None
            return False

    def set_token(self, new_token: str):
        if validate_huggingface_token(new_token):
            self.token = new_token
            logger.info("Token successfully set")
            return True
        else:
            logger.error("Failed to set token")
            return False
