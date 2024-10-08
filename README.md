# AI-Box

This project is about using a docker container to run a large language model in order to be able to generate text. The model is run via using the [LLaMA.cpp HTTP Server](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md) so that the user can interact with it via HTTP requests.

## Installation

### Prerequisites

- Docker
- Docker Compose
- .gguf file with the model weights

### Steps

1. Clone the repository
2. Place the .gguf file in the `model` directory (for example, [bartowski/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/tree/main))
3. Amend the docker-compose.yml file to reflect the name of the .gguf file
4. Run `docker-compose up --build`

## Usage

The server will be running on `localhost:8080`. You can interact with it via HTTP requests. The endpoints are available in the [LLaMA.cpp HTTP Server](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#api-endpoints) documentation.


Example:

```bash
curl --request POST \
    --url http://localhost:8080/completion \
    --header "Content-Type: application/json" \
    --data '{"prompt": "Building a website can be done in 10 simple steps:","n_predict": 128}'
```