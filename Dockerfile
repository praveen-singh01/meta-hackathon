FROM python:3.11-slim

LABEL maintainer="hackathon-team"
LABEL description="Customer Support Simulation – OpenEnv Environment"

WORKDIR /app

# Ensure logs are delivered instantly
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY models.py scenarios.py environment.py graders.py rewards.py inference.py openenv.yaml pyproject.toml uv.lock ./
COPY server ./server

# Expose the default Hugging Face Space port
EXPOSE 7860

# Default command runs the FastAPI server
CMD ["python", "server/app.py"]
