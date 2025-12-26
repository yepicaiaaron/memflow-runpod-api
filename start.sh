#!/bin/bash
set -e

echo "Starting MemFlow RunPod Setup..."

# Install system dependencies
apt-get update -qq
apt-get install -y -qq git wget python3.10 python3-pip ffmpeg git-lfs curl

# Setup Python
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
python3 -m pip install --upgrade pip

# Clone API repository
cd /workspace
if [ ! -d "api" ]; then
  git clone https://github.com/yepicaiaaron/memflow-runpod-api.git api
fi
cd api

# Install dependencies
python3 -m pip install -r requirements.txt

# Clone MemFlow
cd /workspace
if [ ! -d "memflow" ]; then
  git clone https://github.com/KlingTeam/MemFlow.git memflow
fi
cd memflow

# Download models
mkdir -p wan_models/Wan2.1-T2V-1.3B checkpoints
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download KlingTeam/MemFlow --local-dir checkpoints

# Start FastAPI server
cd /workspace/api
echo "Starting FastAPI server on port 8000..."
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
