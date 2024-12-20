# AI-Box

This project uses HuggingFace transformer and Flask for backend to run and deploy a large language model in order to generate log suggestions. The frontend interface is a Visual Studio Code extension using JavaScript.

## Table des matières

- [AI-Box](#ai-box)
  - [Table des matières](#table-des-matières)
  - [Prerequisites](#prerequisites)
  - [Run locally and enable pre-commit](#run-locally-and-enable-pre-commit)
    - [Run pre-commit](#run-pre-commit)
    - [Aditionnal steps to run on GPU locally](#aditionnal-steps-to-run-on-gpu-locally)
  - [Run on Docker](#run-on-docker)
    - [Docker Nvidia setup](#docker-nvidia-setup)
  - [Run tests](#run-tests)
  - [API Documentation](#api-documentation)
    - [API endpoint:`/predict`](#api-endpointpredict)
    - [API endpoint:`/change_model`](#api-endpointchange_model)
    - [API endpoint:`/change_token`](#api-endpointchange_token)
    - [API endpoint:`/model_info`](#api-endpointmodel_info)
  - [Test with Bash](#test-with-bash)

## Prerequisites

- Docker (Docker Desktop)
- Python (tested with 3.11/3.12)

## Run locally and enable pre-commit

1. Open terminal in project root
2. Create python virtual environment
   - Create a python environment with `vscode` : `F1` > `Python : Create Environment...` > `venv`
   - Create a python environment in terminal `python -m venv .venv`
3. Activate Virtual Environement
   - On Windows: `.venv\Scripts\activate`
   - On macOS/Linux: `source .venv/bin/activate`
4. Install dependancies and activate pre-commit:

    ```bash
    pip install -r requirements-dev.txt; pre-commit install
    ```

5. Rename `.env.exemple` to `.env` and add `HF_TOKEN` in the file
6. Start backend: `python app.py`
   - The server will start on [http://localhost:8888](http://localhost:8888)

### Run pre-commit

There are several ways to run pre-commit:

- Commit the code
- Run command: `pre-commit run` (Executed on changed files)
- Run command: `pre-commit run --all-files` (Force execute on all files)

### Aditionnal steps to run on GPU locally

1. Install CUDA on your local Machine : [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
2. Install Torch-Cuda : [Start Locally - PyTorch](https://pytorch.org/get-started/locally/)
3. Start backend: `python app.py`
   - The server will start on [http://localhost:8888](http://localhost:8888)

## Run on Docker

*If a compatible CUDA device is detected, the container will be executed on the GPU.*

1. Open Docker Desktop
2. Create and Run container `docker-compose up --build`
   - The server will start on [http://localhost:8888](http://localhost:8888)

### Docker Nvidia setup

1. Install Nvidia Cuda

    ```bash
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-6
    ```

2. Install Nvidia Drivers

    ```bash
    sudo apt-get install -y nvidia-open
    sudo apt-get install -y cuda-drivers
    ```

3. Config nvidia-docker runtime

    ```bash
    sudo apt-get install -y nvidia-docker2
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

4. Look at apps running on GPU in this commmand :

    ```bahs
    nvidia-smi
    ```

5. In Docker Desktop for windows, enable Ubuntu WSL integration `Options > Resources > WSL Integration`.

## Run tests

1. Open terminal in project root
2. install dev dependancies: `pip install -r requirements-dev.txt`
3. Run tests: `pytest`

## API Documentation

Once the application is running, you can find the documentation at [http://localhost:8888/docs](http://localhost:8888/docs)

### API endpoint:`/predict`

This endpoint allows the client to ask a question to the model. It accepts a JSON payload with a `prompt` key for the user input query. The `max_new_tokens` key specifies the maximum number of new tokens the model should generate in response to the input prompt, if not specified, 128 by default.

**Endpoint:** `/predict`
**Method:** `POST`
**Request body:**

```json
{
    "prompt": "USER_PROMPT",
    "max_new_tokens": "MAX_TOKENS"
}
```

**Response body:**

```json
{
    "content": "MODEL_RESPONSE",
}
```

- `content` (str): Model response.

### API endpoint:`/change_model`

This endpoint allows the client to update the current model by specifying
the model ID. It accepts a JSON payload with a `hf_model_id` key and responds with the operation status.

**Endpoint:** `/change_model`
**Method:** `POST`
**Request body:**

```json
{
    "hf_model_id": "MODEL_ID"
}
```

**Response body:**

```json
{
    "completed": boolean,
    "model_name": "MODEL_ID"
}
```

- `completed` (bool): True if the token change was successful, False otherwise.
- `model_name` (str): The name / id of the model currently in use (for confirmation).

### API endpoint:`/change_token`

This endpoint allows the client to update the current huggingface token by specifying a token. It accepts a JSON payload with a `token` key and  responds with the operation status.

**Endpoint:** `/change_token`
**Method:** `POST`
**Request body:**

```json
{
    "token": "HUGGINGFACE_TOKEN",
}
```

**Response body:**

```json
{
    "completed": boolean,
}
```

- `completed` (bool): True if the token change was successful, False otherwise.

### API endpoint:`/model_info`

This endpoint allows the client to get basic information about the active model in the application.

**Endpoint:** `/model_info`
**Method:** `GET`
**Response body:**

```json
{
    "model_name": "MODEL_ID",
    "device": "DEVICE"
}
```

- `model_name` (str): The name of the model currently in use.
- `device` (str): Information about the device executing the model (cpu) or (cuda)

## Test with Bash

```bash
curl --request POST \
    --url http://localhost:8888/predict \
    --header "Content-Type: application/json" \
    --data '{"prompt": "Building a website can be done in 10 simple steps:","max_new_tokens": 128}'
```
