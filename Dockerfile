# Use an official CUDA runtime as a base image
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose the port the app runs on
EXPOSE 8888

# Run the Flask app
CMD ["python3", "run_model.py"]