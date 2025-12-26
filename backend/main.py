"""FastAPI backend for MemFlow video generation."""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys
import os
import torch

# Add MemFlow to path
sys.path.append('/app/memflow')

app = FastAPI(title="MemFlow API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model initialization (lazy loading)
model = None

class VideoRequest(BaseModel):
    prompt: str
    num_frames: int = 121
    fps: int = 24
    guidance_scale: float = 7.5


class InteractiveVideoRequest(BaseModel):
    prompts: list[str]
    num_frames_per_prompt: int = 121
    fps: int = 24


def load_model():
    """Lazy load MemFlow model."""
    global model
    if model is None:
        from models import MemFlowModel
        model = MemFlowModel(
            checkpoint_path="/app/checkpoints",
            model_path="/app/wan_models/Wan2.1-T2V-1.3B"
        )
    return model


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "MemFlow API is running",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.post("/generate")
async def generate_video(request: VideoRequest):
    """Generate video from single text prompt."""
    try:
        model_instance = load_model()
        video_path = await model_instance.generate(
            prompt=request.prompt,
            num_frames=request.num_frames,
            fps=request.fps,
            guidance_scale=request.guidance_scale
        )
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=f"memflow_{request.prompt[:30]}.mp4"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate-interactive")
async def generate_interactive_video(request: InteractiveVideoRequest):
    """Generate long video from multiple prompts."""
    try:
        model_instance = load_model()
        video_path = await model_instance.generate_interactive(
            prompts=request.prompts,
            num_frames=request.num_frames_per_prompt,
            fps=request.fps
        )
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename="memflow_interactive.mp4"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interactive generation failed: {str(e)}")


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio to text using Whisper."""
    try:
        import whisper
        import tempfile
        
        # Load Whisper model
        whisper_model = whisper.load_model("base")
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
        
        # Transcribe
        result = whisper_model.transcribe(temp_audio_path)
        transcript = result["text"]
        
        # Cleanup
        os.remove(temp_audio_path)
        
        return {"text": transcript.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
