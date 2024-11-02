import logging
import os

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


def validate_huggingface_token(hf_token: str) -> bool:
    headers = {"Authorization": f"Bearer {hf_token}"}
    response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
    return response.status_code == 200


class ModelManager:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.token = None
        self.tokenizer = None
        self.device = None
        self.model_name = None

    def create_model(self, model_name: str, hf_token: str):
        try:
            # Check if the model is already in the model directory, otherwise download it
            if not os.path.exists(os.path.join(self.model_dir, model_name)):
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=self.model_dir,
                    token=hf_token,
                )
                # Add a new [PAD] hf_token
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=self.model_dir,
                    token=hf_token,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_dir, trust_remote_code=True, hf_token=hf_token
                )
                # Add a new [PAD] hf_token
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_dir,
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

    def set_model_name(self, model_name: str):
        response = self.create_model(model_name, self.token)
        if response is not None:
            self.model, self.tokenizer, self.device = response
            self.model_name = model_name
            logger.info(f"Model set to {model_name}")
            return True
        else:
            logger.error(f"Failed to set model to {model_name}")
            return False

    def set_token(self, new_token: str):
        validate = validate_huggingface_token(new_token)
        response = self.create_model(self.model_name, new_token)
        if validate and response is not None:
            self.model, self.tokenizer, self.device = response
            self.token = new_token
            logger.info("Token updated successfully")
            return True
        else:
            logger.error("Failed to update hf_token")
            return False

    def init_model(self):
        # Model details
        # meta-llama/Llama-3.2-1B-Instruct
        # Qwen/Qwen2.5-Coder-1.5B-Instruct

        self.model_name = "meta-llama/Llama-3.2-1B-Instruct"
        self.token = "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"
        self.set_model_name(self.model_name)
        logger.info(f"Device: {self.device}")
