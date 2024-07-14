# Base image
FROM python:3.11-slim as base

WORKDIR /app

# Copy the requirements file and install dependencies
COPY ./requirements.txt .

# Update pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Torch stage
FROM base as simple-cnn-torch
ENTRYPOINT ["python", "simple-cnn/torch/train.py"]

# Jax stage
FROM base as simple-cnn-jax
ENTRYPOINT ["python", "simple-cnn/jax/train.py"]

FROM base as simple-eq-torch
ENTRYPOINT ["python", "simple-eq/torch/train.py"]

FROM base as simple-eq-jax
ENTRYPOINT ["python", "simple-eq/jax/train.py"]
