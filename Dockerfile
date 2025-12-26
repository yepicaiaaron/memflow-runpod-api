# MemFlow RunPod Deployment
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    git-lfs \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Clone MemFlow repository
RUN git clone https://github.com/KlingTeam/MemFlow /app/memflow

# Download model checkpoints
RUN pip3 install "huggingface_hub[cli]" && \
    huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir /app/wan_models/Wan2.1-T2V-1.3B && \
    huggingface-cli download KlingTeam/MemFlow --local-dir /app/checkpoints

# Copy application code
COPY backend /app/backend
COPY frontend/build /app/frontend/build

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
