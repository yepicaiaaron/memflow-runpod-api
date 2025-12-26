# MemFlow RunPod Deployment Guide

This guide walks you through deploying the MemFlow text-to-video API on RunPod.

## Prerequisites

- Docker installed locally
- Docker Hub account
- RunPod account with sufficient credits
- 80GB+ GPU (A100, H100, or equivalent)

## Step 1: Build Docker Image

```bash
# Clone the repository
git clone https://github.com/yepicaiaaron/memflow-runpod-api.git
cd memflow-runpod-api

# Build the Docker image
docker build -t your-dockerhub-username/memflow-runpod:latest .
```

**Note**: Building may take 30-60 minutes due to model downloads.

## Step 2: Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Push the image
docker push your-dockerhub-username/memflow-runpod:latest
```

## Step 3: Deploy on RunPod

### 3.1 Create Pod Template (One-time)

1. Go to RunPod Console > Templates
2. Click "New Template"
3. Configure:
   - **Template Name**: MemFlow API
   - **Container Image**: your-dockerhub-username/memflow-runpod:latest
   - **Expose HTTP Ports**: 8000
   - **Container Disk**: 100 GB minimum
   - **Environment Variables** (optional):
     - `MODEL_PATH=/app/memflow`
     - `CHECKPOINT_PATH=/app/checkpoints`

### 3.2 Deploy Pod

1. Go to **Pods** > **+ Deploy**
2. Select GPU: **H100 PCIe** (80GB) or **A100** (80GB)
3. GPU Count: **1**
4. Select your **MemFlow API** template
5. Configure storage:
   - **Container Disk**: 100 GB
   - **Volume Disk**: Optional (for persistent storage)
6. Click **Deploy**

### 3.3 Wait for Pod to Start

The pod will:
1. Pull the Docker image (~15-20 GB)
2. Start the container
3. Load models into VRAM (~60-70 GB)
4. Start FastAPI server on port 8000

**Startup time**: 5-10 minutes

## Step 4: Test the API

### 4.1 Get Pod URL

Once running, RunPod will provide a URL like:
```
https://YOUR-POD-ID-8000.proxy.runpod.net
```

### 4.2 Test Health Endpoint

```bash
curl https://YOUR-POD-ID-8000.proxy.runpod.net/health
```

Expected response:
```json
{"status": "healthy", "model_loaded": true}
```

### 4.3 Test Video Generation

```bash
curl -X POST https://YOUR-POD-ID-8000.proxy.runpod.net/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A serene lake at sunset"}'
```

Response will include:
```json
{
  "video_url": "/outputs/video_timestamp.mp4",
  "prompt": "A serene lake at sunset",
  "generation_time": 45.2
}
```

### 4.4 Test Voice Transcription

```bash
curl -X POST https://YOUR-POD-ID-8000.proxy.runpod.net/transcribe \
  -F "audio=@your_audio.wav"
```

## Step 5: Frontend Setup (Optional)

The React frontend can be:

**Option A: Build and include in Docker image**
```bash
cd frontend
npm install
npm run build
# Rebuild Docker image with frontend/build included
```

**Option B: Deploy separately on Vercel/Netlify**
```bash
cd frontend
npm install
# Set API_URL environment variable to your RunPod URL
VEREL_API_URL=https://YOUR-POD-ID-8000.proxy.runpod.net
npm run build
# Deploy to Vercel
```

## Monitoring

### View Logs
```bash
# In RunPod console, click on your pod > Logs
```

### Check GPU Usage
```bash
# Connect to pod terminal and run:
nvidia-smi
```

## Cost Estimation

- **H100 PCIe**: ~$2.39/hr on-demand
- **A100 80GB**: ~$1.89/hr on-demand
- Use Spot instances for 50-70% savings

## Troubleshooting

### Pod fails to start
- Check Docker image exists and is accessible
- Verify 80GB+ GPU is selected
- Check RunPod logs for errors

### Model loading fails
- Ensure sufficient disk space (100GB+)
- Verify Hugging Face credentials if using gated models

### API timeout
- First generation takes longer (model warmup)
- Increase timeout to 60s for initial requests
