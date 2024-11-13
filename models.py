from pydantic import BaseModel, field_validator


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
