# MemFlow RunPod API

Deploy MemFlow text-to-video generation model on RunPod with FastAPI backend and React frontend supporting text and voice input.

## 🎯 Features

- **Text-to-Video Generation**: Generate high-quality videos from text prompts using MemFlow
- **Voice Input Support**: Record voice notes that are transcribed to text prompts
- **Interactive Long Video**: Create extended video narratives with multiple prompts
- **Real-time Inference**: 18.7 FPS on H100 GPU
- **FastAPI Backend**: RESTful API with async support
- **React Frontend**: Modern UI with voice recording capabilities
- **RunPod Deployment**: Optimized Docker container for GPU instances

## 📁 Project Structure

```
memflow-runpod-api/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── models.py            # MemFlow model wrapper
│   ├── utils.py             # Helper functions
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.js          # Main React component
│   │   ├── components/
│   │   │   ├── VideoGenerator.js
│   │   │   ├── VoiceRecorder.js
│   │   │   └── VideoPlayer.js
│   │   └── index.js
│   ├── package.json
│   └── public/
├── Dockerfile              # RunPod deployment
├── requirements.txt        # Backend dependencies
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- NVIDIA GPU with 80GB VRAM (A100/H100)
- CUDA 12.4+
- Docker
- RunPod account

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/yepicaiaaron/memflow-runpod-api.git
cd memflow-runpod-api
```

2. **Set up backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

3. **Set up frontend**
```bash
cd frontend
npm install
npm start
```

### RunPod Deployment

1. **Build Docker image**
```bash
docker build -t memflow-runpod:latest .
```

2. **Push to Docker Hub**
```bash
docker tag memflow-runpod:latest yourusername/memflow-runpod:latest
docker push yourusername/memflow-runpod:latest
```

3. **Deploy on RunPod**
- Go to RunPod.io
- Create new pod with A100/H100 GPU
- Use custom Docker image: `yourusername/memflow-runpod:latest`
- Expose port 8000
- Start pod

## 📝 Backend Implementation

### backend/main.py

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys
import os
sys.path.append('/app/memflow')
from models import MemFlowModel
import whisper
import tempfile

app = FastAPI(title="MemFlow API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
model = MemFlowModel(
    checkpoint_path="/app/checkpoints",
    model_path="/app/wan_models/Wan2.1-T2V-1.3B"
)

# Initialize Whisper for transcription
whisper_model = whisper.load_model("base")

class VideoRequest(BaseModel):
    prompt: str
    num_frames: int = 121
    fps: int = 24
    guidance_scale: float = 7.5

class InteractiveVideoRequest(BaseModel):
    prompts: list[str]
    num_frames_per_prompt: int = 121
    fps: int = 24

@app.get("/")
async def root():
    return {"message": "MemFlow API is running", "status": "healthy"}

@app.post("/generate")
async def generate_video(request: VideoRequest):
    """Generate video from single text prompt"""
    try:
        video_path = await model.generate(
            prompt=request.prompt,
            num_frames=request.num_frames,
            fps=request.fps,
            guidance_scale=request.guidance_scale
        )
        return FileResponse(
            video_path, 
            media_type="video/mp4",
            filename="generated_video.mp4"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-interactive")
async def generate_interactive_video(request: InteractiveVideoRequest):
    """Generate long video from multiple prompts"""
    try:
        video_path = await model.generate_interactive(
            prompts=request.prompts,
            num_frames=request.num_frames_per_prompt,
            fps=request.fps
        )
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename="interactive_video.mp4"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio to text using Whisper"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
        
        # Transcribe with Whisper
        result = whisper_model.transcribe(temp_audio_path)
        transcript = result["text"]
        
        # Clean up temp file
        os.remove(temp_audio_path)
        
        return {"text": transcript.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model.is_loaded(),
        "gpu_available": torch.cuda.is_available()
    }
```

### backend/models.py

```python
import torch
import sys
import os
from pathlib import Path
sys.path.append('/app/memflow')

class MemFlowModel:
    def __init__(self, checkpoint_path, model_path):
        self.checkpoint_path = checkpoint_path
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load MemFlow model"""
        try:
            from inference import load_model, inference
            self.model = load_model(
                checkpoint=self.checkpoint_path,
                base_model=self.model_path,
                device=self.device
            )
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def is_loaded(self):
        return self.model is not None
    
    async def generate(self, prompt, num_frames=121, fps=24, guidance_scale=7.5):
        """Generate video from single prompt"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        with torch.no_grad():
            # Run inference
            video = self.model.inference(
                prompts=[prompt],
                num_frames=num_frames,
                fps=fps,
                guidance_scale=guidance_scale
            )
        
        # Save video
        output_path = f"/app/outputs/video_{torch.randint(0, 10000, (1,)).item()}.mp4"
        os.makedirs("/app/outputs", exist_ok=True)
        self.save_video(video, output_path, fps)
        
        return output_path
    
    async def generate_interactive(self, prompts, num_frames=121, fps=24):
        """Generate long video from multiple prompts"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        with torch.no_grad():
            video = self.model.interactive_inference(
                prompts=prompts,
                num_frames=num_frames,
                fps=fps
            )
        
        output_path = f"/app/outputs/interactive_{torch.randint(0, 10000, (1,)).item()}.mp4"
        os.makedirs("/app/outputs", exist_ok=True)
        self.save_video(video, output_path, fps)
        
        return output_path
    
    def save_video(self, video_tensor, output_path, fps):
        """Save video tensor to file"""
        import torchvision
        torchvision.io.write_video(
            output_path,
            video_tensor,
            fps=fps
        )
```

### backend/requirements.txt

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.8.0
torchvision==0.23.0
transformers==4.36.0
diffusers==0.25.0
huggingface-hub[cli]
openai-whisper
python-multipart
pillow
numpy
accelerate
```

## 🎨 Frontend Implementation

### frontend/src/App.js

```jsx
import React, { useState } from 'react';
import VideoGenerator from './components/VideoGenerator';
import VoiceRecorder from './components/VoiceRecorder';
import VideoPlayer from './components/VideoPlayer';

function App() {
  const [prompt, setPrompt] = useState('');
  const [videoUrl, setVideoUrl] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    setLoading(true);
    const response = await fetch('http://localhost:8000/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });
    const blob = await response.blob();
    setVideoUrl(URL.createObjectURL(blob));
    setLoading(false);
  };

  const handleVoiceInput = async (audioBlob) => {
    const formData = new FormData();
    formData.append('file', audioBlob);
    const response = await fetch('http://localhost:8000/transcribe', {
      method: 'POST',
      body: formData
    });
    const data = await response.json();
    setPrompt(data.text);
  };

  return (
    <div className="App">
      <h1>MemFlow Video Generator</h1>
      <VideoGenerator prompt={prompt} setPrompt={setPrompt} onGenerate={handleGenerate} loading={loading} />
      <VoiceRecorder onRecordingComplete={handleVoiceInput} />
      {videoUrl && <VideoPlayer url={videoUrl} />}
    </div>
  );
}

export default App;
```

### Voice Recorder Component

```jsx
const VoiceRecorder = ({ onRecordingComplete }) => {
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef(null);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new MediaRecorder(stream);
    const chunks = [];
    mediaRecorderRef.current.ondataavailable = (e) => chunks.push(e.data);
    mediaRecorderRef.current.onstop = () => {
      onRecordingComplete(new Blob(chunks, { type: 'audio/webm' }));
    };
    mediaRecorderRef.current.start();
    setRecording(true);
  };

  return (
    <button onClick={recording ? stopRecording : startRecording}>
      {recording ? 'Stop' : 'Record Voice'}
    </button>
  );
};
```

## 📦 Dependencies

### requirements.txt
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.8.0
torchvision==0.23.0
openai-whisper
huggingface-hub[cli]
flash-attn
```

### package.json
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  }
}
```

## 📊 Performance

- **GPU**: H100 (80GB VRAM)
- **Speed**: 18.7 FPS
- **Memory**: ~60GB VRAM
- **Latency**: ~6.5s for 121 frames

## 🎯 API Endpoints

- `POST /generate` - Generate video from text
- `POST /transcribe` - Transcribe audio to text
- `GET /health` - Health check

## 🙏 Acknowledgments

- [MemFlow](https://github.com/KlingTeam/MemFlow)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [RunPod](https://runpod.io)

## 📄 License

MIT
