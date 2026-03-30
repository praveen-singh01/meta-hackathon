FROM python:3.11-slim

LABEL maintainer="hackathon-team"
LABEL description="Customer Support Simulation – OpenEnv Environment"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY models.py scenarios.py environment.py graders.py rewards.py inference.py openenv.yaml ./

# Default command runs the inference agent
CMD ["python", "inference.py"]
