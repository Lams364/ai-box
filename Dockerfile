# Use an official CUDA runtime as a base image
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

# Set the working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies within the virtual environment
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose the port the app runs on
EXPOSE 8888

# Run the Flask app using the virtual environment's Python
CMD ["/opt/venv/bin/python", "app.py"]
