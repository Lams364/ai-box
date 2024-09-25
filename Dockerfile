FROM python:3.9-slim

# Install required dependencies
RUN pip install flask torch transformers

# Copy your app into the container
COPY app.py /app.py

# Expose the port Flask will run on
EXPOSE 5000

# Run the Flask app
CMD ["python", "/app.py"]
