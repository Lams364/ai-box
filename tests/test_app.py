import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

import app as app_file
from app import app

# Load environment variables from .env file
load_dotenv()


client = TestClient(app)


def test_predict():
    response = client.post(
        "/predict", json={"prompt": "Hello, world!", "max_new_tokens": 5}
    )
    assert response.status_code == 200
    assert "content" in response.json()


def test_generate_log_advice():

    with open("tests/escaped_code.text", "r") as file:
        escaped_code = file.read()

    response = client.post(
        "/generate/log-advice", json={"prompt": escaped_code, "max_new_tokens": 10}
    )
    assert response.status_code == 200
    assert "content" in response.json()


def test_predict_no_prompt():
    response = client.post("/predict", json={"max_new_tokens": 5})
    assert response.status_code == 400
    assert response.json() == {"errors": ["prompt: Field required"]}


def test_change_model():
    response = client.post(
        "/change_model", json={"hf_model_id": "meta-llama/Llama-3.2-1B-Instruct"}
    )
    assert response.status_code == 200
    assert response.json()["completed"] is True
    assert response.json()["model_name"] == "meta-llama/Llama-3.2-1B-Instruct"


def test_change_model_no_id():
    response = client.post("/change_model", json={})
    assert response.status_code == 400
    assert response.json() == {"errors": ["hf_model_id: Field required"]}


def test_change_token():
    response = client.post(
        "/change_token", json={"hf_token": "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"}
    )
    assert response.status_code == 200
    assert response.json()["completed"] is True


def test_change_token_no_token():
    response = client.post("/change_token", json={})
    assert response.status_code == 400
    assert response.json() == {"errors": ["hf_token: Field required"]}


def test_model_info():
    response = client.get("/model_info")
    assert response.status_code == 200
    assert "model_name" in response.json()
    assert "device" in response.json()
    assert "model_dir" in response.json()
