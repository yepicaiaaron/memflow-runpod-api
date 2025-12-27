"""FastAPI backend for MemFlow video generation with direct Python integration."""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys
import os
import torch
from omegaconf import OmegaConf
import logging
from pathlib import Path
import uuid
from typing import Optional
import tempfile
from einops import rearrange
from torchvision.io import write_video

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

# Global pipeline variable
pipeline = None
config = None

class VideoRequest(BaseModel):
    prompt: str
    num_frames: int = 121
    fps: int = 24
    guidance_scale: float = 7.5

@app.on_event("startup")
async def startup_event():
    """Initialize MemFlow pipeline on startup."""
    global pipeline, config
    try:
        logger.info("Initializing MemFlow pipeline...")
        
        # Import MemFlow modules
        from pipeline import CausalInferencePipeline
        
        # Create config programmatically
        config = OmegaConf.create({
            'denoising_step_list': [1000, 750, 500, 250],
            'warp_denoising_step': True,
            'num_frame_per_block': 3,
            'model_name': 'Wan2.1-T2V-1.3B',
            'model_kwargs': {
                'local_attn_size': 12,
                'timestep_shift': 5.0,
                'sink_size': 3,
                'bank_size': 3,
                'record_interval': 3,
                'SMA': False
            },
            'num_output_frames': 120,
            'use_ema': False,
            'seed': 0,
            'num_samples': 1,
            'global_sink': True,
            'context_noise': 0,
            'generator_ckpt': '/workspace/memflow/checkpoints/base.pt',
            'lora_ckpt': '/workspace/memflow/checkpoints/lora.pt',
            'adapter': {
                'type': 'lora',
                'rank': 256,
                'alpha': 256,
                'dropout': 0.0,
                'dtype': 'bfloat16',
                'verbose': False
            },
            'distributed': False
        })
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize pipeline
        pipeline = CausalInferencePipeline(config, device=device)
        
        # Load generator checkpoint
        if config.generator_ckpt and Path(config.generator_ckpt).exists():
            logger.info(f"Loading checkpoint from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict or "generator_ema" in state_dict:
                raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
            elif "model" in state_dict:
                raw_gen_state_dict = state_dict["model"]
            else:
                logger.warning("Generator state dict not found in checkpoint")
                raw_gen_state_dict = None
            
            if raw_gen_state_dict:
                pipeline.generator.load_state_dict(raw_gen_state_dict, strict=False)
                logger.info("Generator checkpoint loaded")
        
        # Load LoRA if available
        if hasattr(config, 'adapter') and config.adapter:
            try:
                from utils.lora_utils import configure_lora_for_model
                import peft
                
                logger.info("Applying LoRA to generator")
                pipeline.generator.model = configure_lora_for_model(
                    pipeline.generator.model,
                    model_name="generator",
                    lora_config=config.adapter,
                    is_main_process=True
                )
                
                if config.lora_ckpt and Path(config.lora_ckpt).exists():
                    logger.info(f"Loading LoRA checkpoint from {config.lora_ckpt}")
                    lora_checkpoint = torch.load(config.lora_ckpt, map_location="cpu")
                    if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                        peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
                    else:
                        peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
                    logger.info("LoRA weights loaded")
                    pipeline.is_lora_enabled = True
            except Exception as e:
                logger.warning(f"LoRA loading failed: {e}")
                pipeline.is_lora_enabled = False
        
        # Move to appropriate dtype and device
        pipeline = pipeline.to(dtype=torch.bfloat16)
        pipeline.generator.to(device=device)
        pipeline.vae.to(device=device)
        
        logger.info("MemFlow pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MemFlow: {e}")
        import traceback
        logger.error(traceback.format_exc())

@app.get("/")
async def root():
    return {"message": "MemFlow API is running", "version": "1.0.0"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    memflow_available = pipeline is not None
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    
    return {
        "status": "healthy" if memflow_available else "initializing",
        "memflow_available": memflow_available,
        "config_exists": config is not None,
        "gpu_available": gpu_available,
        "gpu_count": gpu_count
    }

@app.post("/generate")
async def generate_video(request: VideoRequest):
    """Generate video from text prompt."""
    global pipeline, config
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="MemFlow pipeline not initialized")
    
    try:
        logger.info(f"Generating video for prompt: {request.prompt}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Prepare prompt
        prompts = [request.prompt]
        
        # Generate noise
        sampled_noise = torch.randn(
            [1, config.num_output_frames, 16, 60, 104], 
            device=device, 
            dtype=torch.bfloat16
        )
        
        # Generate video
        logger.info("Running inference...")
        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            low_memory=True,
            profile=False
        )
        
        # Process video
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        video_output = 255.0 * current_video
        
        # Save video to temporary file
        video_id = str(uuid.uuid4())
        output_dir = Path("/workspace/api/videos")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_id}.mp4"
        
        write_video(str(output_path), video_output[0], fps=request.fps)
        
        # Clear cache
        if hasattr(pipeline.vae, 'model') and hasattr(pipeline.vae.model, 'clear_cache'):
            pipeline.vae.model.clear_cache()
        
        logger.info(f"Video generated successfully: {output_path}")
        
        return {
            "video_id": video_id,
            "status": "completed",
            "prompt": request.prompt,
            "num_frames": config.num_output_frames,
            "fps": request.fps
        }
        
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video/{video_id}")
async def get_video(video_id: str):
    """Retrieve generated video."""
    video_path = Path(f"/workspace/api/videos/{video_id}.mp4")
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{video_id}.mp4"
    )

@app.get("/videos")
async def list_videos():
    """List all generated videos."""
    video_dir = Path("/workspace/api/videos")
    if not video_dir.exists():
        return {"videos": []}
    
    videos = []
    for video_file in video_dir.glob("*.mp4"):
        videos.append({
            "video_id": video_file.stem,
            "filename": video_file.name,
            "size": video_file.stat().st_size
        })
    
    return {"videos": videos}
