"""FastAPI backend for MemFlow video generation using subprocess execution."""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys
import os
import torch
import subprocess
import json
import uuid
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add MemFlow to path
sys.path.append('/workspace/memflow')

app = FastAPI(title="MemFlow API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output directory for videos
OUTPUT_DIR = Path("/workspace/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# MemFlow paths
MEMFLOW_DIR = Path("/workspace/memflow")
CONFIG_PATH = MEMFLOW_DIR / "configs" / "inference.yaml"

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
    return {"message": "MemFlow API is running", "version": "1.0.0"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "memflow_available": MEMFLOW_DIR.exists(),
        "config_exists": CONFIG_PATH.exists(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

@app.post("/generate")
async def generate_video(request: VideoRequest):
    """Generate video from single text prompt using MemFlow subprocess."""
    try:
        # Create unique output filename
        video_id = str(uuid.uuid4())
        output_file = OUTPUT_DIR / f"{video_id}.mp4"
        
        # Create temporary prompt file
        prompt_file = OUTPUT_DIR / f"{video_id}_prompt.txt"
        with open(prompt_file, 'w') as f:
            f.write(request.prompt)
        
        logger.info(f"Generating video for prompt: {request.prompt}")
        
        # Build command to run MemFlow inference
        cmd = [
            sys.executable, str(MEMFLOW_DIR / "inference.py"),
            "--config_path", str(CONFIG_PATH),
            "--prompt", request.prompt,
            "--output_folder", str(OUTPUT_DIR),
            "--num_samples", "1",
            "--guidance_scale", str(request.guidance_scale)
        ]
        
        # Execute MemFlow inference
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.run(
            cmd,
            
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if process.returncode != 0:
            logger.error(f"MemFlow error: {process.stderr}")
            raise HTTPException(
                status_code=500,
                detail=f"Video generation failed: {process.stderr}"
            )
        
        # Find generated video file
        # MemFlow saves videos with pattern: output_folder/regular_0.mp4
        generated_files = list(OUTPUT_DIR.glob("*.mp4"))
        if not generated_files:
            raise HTTPException(
                status_code=500,
                detail="Video was generated but file not found"
            )
        
        # Get the most recently created video
        latest_video = max(generated_files, key=lambda p: p.stat().st_ctime)
        
        # Rename to our UUID
        latest_video.rename(output_file)
        
        # Cleanup prompt file
        prompt_file.unlink(missing_ok=True)
        
        logger.info(f"Video generated successfully: {output_file}")
        
        return {
            "video_id": video_id,
            "status": "success",
            "video_url": f"/video/{video_id}",
            "prompt": request.prompt,
            "duration_seconds": request.num_frames / request.fps
        }
        
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Video generation timeout (>5 minutes)"
        )
    except Exception as e:
        logger.error(f"Error generating video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-interactive")
async def generate_interactive_video(request: InteractiveVideoRequest):
    """Generate interactive long video from multiple prompts."""
    try:
        video_id = str(uuid.uuid4())
        
        # Create JSONL file with prompts for interactive generation
        prompts_file = OUTPUT_DIR / f"{video_id}_prompts.jsonl"
        with open(prompts_file, 'w') as f:
            for i, prompt in enumerate(request.prompts):
                json.dump({"text": prompt, "index": i}, f)
                f.write('\n')
        
        logger.info(f"Generating interactive video with {len(request.prompts)} prompts")
        
        # Run interactive inference
        cmd = [
            sys.executable, str(MEMFLOW_DIR / "interactive_inference.py"),
            "--config_path", str(MEMFLOW_DIR / "configs" / "interactive_inference.yaml"),
            "--extended_prompt_path", str(prompts_file),
            "--output_folder", str(OUTPUT_DIR)
        
        process = subprocess.run(
            cmd,
            cwd=str(MEMFLOW_DIR),
                    ]
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for long video
        )
        
        if process.returncode != 0:
            logger.error(f"Interactive generation error: {process.stderr}")
            raise HTTPException(
                status_code=500,
                detail=f"Interactive video generation failed: {process.stderr}"
            )
        
        # Find and rename generated video
        generated_files = list(OUTPUT_DIR.glob("*.mp4"))
        if not generated_files:
            raise HTTPException(
                status_code=500,
                detail="Interactive video was generated but file not found"
            )
        
        latest_video = max(generated_files, key=lambda p: p.stat().st_ctime)
        output_file = OUTPUT_DIR / f"{video_id}.mp4"
        latest_video.rename(output_file)
        
        prompts_file.unlink(missing_ok=True)
        
        return {
            "video_id": video_id,
            "status": "success",
            "video_url": f"/video/{video_id}",
            "num_prompts": len(request.prompts),
            "prompts": request.prompts
        }
        
    except Exception as e:
        logger.error(f"Error in interactive generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio to text using Whisper."""
    try:
        import whisper
        
        # Save uploaded file
        audio_path = OUTPUT_DIR / f"temp_{uuid.uuid4()}.{file.filename.split('.')[-1]}"
        with open(audio_path, 'wb') as f:
            f.write(await file.read())
        
        # Load Whisper model
        model = whisper.load_model("base")
        
        # Transcribe
        result = model.transcribe(str(audio_path))
        
        # Cleanup
        audio_path.unlink(missing_ok=True)
        
        return {
            "text": result["text"],
            "language": result.get("language", "unknown")
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video/{video_id}")
async def get_video(video_id: str):
    """Retrieve generated video by ID."""
    video_path = OUTPUT_DIR / f"{video_id}.mp4"
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"memflow_{video_id}.mp4"
    )

@app.get("/videos")
async def list_videos():
    """List all generated videos."""
    videos = []
    for video_file in OUTPUT_DIR.glob("*.mp4"):
        videos.append({
            "video_id": video_file.stem,
            "filename": video_file.name,
            "size_mb": video_file.stat().st_size / (1024 * 1024),
            "created": video_file.stat().st_ctime
        })
    return {"videos": videos, "count": len(videos)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
