"""MemFlow model wrapper for video generation."""
import torch
import sys
import os
from pathlib import Path
import numpy as np

sys.path.append('/app/memflow')

class MemFlowModel:
    """Wrapper class for MemFlow text-to-video model."""
    
    def __init__(self, checkpoint_path: str, model_path: str):
        """Initialize MemFlow model.
        
        Args:
            checkpoint_path: Path to MemFlow checkpoints
            model_path: Path to base Wan2.1 model
        """
        self.checkpoint_path = checkpoint_path
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.pipeline = None
        print(f"Initializing MemFlow on {self.device}")
        self.load_model()
    
    def load_model(self):
        """Load MemFlow model and pipeline."""
        try:
            # Import MemFlow modules
            from memflow.inference import load_model as load_memflow
            
            # Load the model
            self.model = load_memflow(
                checkpoint_dir=self.checkpoint_path,
                model_path=self.model_path,
                device=self.device
            )
            
            print(f"✓ MemFlow model loaded successfully")
            print(f"✓ Device: {self.device}")
            if torch.cuda.is_available():
                print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
                print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                
        except Exception as e:
            print(f"Error loading MemFlow model: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    async def generate(self, 
                      prompt: str, 
                      num_frames: int = 121, 
                      fps: int = 24,
                      guidance_scale: float = 7.5) -> str:
        """Generate video from single text prompt.
        
        Args:
            prompt: Text description for video generation
            num_frames: Number of frames to generate
            fps: Frames per second
            guidance_scale: Guidance scale for generation
            
        Returns:
            Path to generated video file
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        print(f"Generating video: '{prompt}'")
        print(f"Frames: {num_frames}, FPS: {fps}, Guidance: {guidance_scale}")
        
        try:
            with torch.no_grad():
                # Run MemFlow inference
                video_frames = self.model.generate(
                    prompt=prompt,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale
                )
            
            # Save video
            output_dir = Path("/app/outputs")
            output_dir.mkdir(exist_ok=True)
            
            output_path = output_dir / f"video_{torch.randint(0, 100000, (1,)).item()}.mp4"
            self._save_video(video_frames, str(output_path), fps)
            
            print(f"✓ Video saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"Generation failed: {e}")
            raise
    
    async def generate_interactive(self,
                                  prompts: list[str],
                                  num_frames: int = 121,
                                  fps: int = 24) -> str:
        """Generate long video from multiple prompts.
        
        Args:
            prompts: List of text prompts for each segment
            num_frames: Frames per prompt segment
            fps: Frames per second
            
        Returns:
            Path to generated video file
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        print(f"Generating interactive video with {len(prompts)} prompts")
        
        try:
            with torch.no_grad():
                # Run interactive MemFlow inference
                video_frames = self.model.generate_interactive(
                    prompts=prompts,
                    num_frames_per_segment=num_frames
                )
            
            # Save video
            output_dir = Path("/app/outputs")
            output_dir.mkdir(exist_ok=True)
            
            output_path = output_dir / f"interactive_{torch.randint(0, 100000, (1,)).item()}.mp4"
            self._save_video(video_frames, str(output_path), fps)
            
            print(f"✓ Interactive video saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"Interactive generation failed: {e}")
            raise
    
    def _save_video(self, frames, output_path: str, fps: int):
        """Save video frames to file.
        
        Args:
            frames: Video frames tensor or numpy array
            output_path: Output file path
            fps: Frames per second
        """
        try:
            import torchvision
            from torchvision.io import write_video
            
            # Convert to appropriate format if needed
            if isinstance(frames, torch.Tensor):
                # Ensure correct shape: (T, H, W, C)
                if frames.dim() == 4 and frames.shape[1] == 3:
                    frames = frames.permute(0, 2, 3, 1)
                
                # Convert to uint8 if needed
                if frames.dtype == torch.float32:
                    frames = (frames * 255).clamp(0, 255).byte()
            
            # Write video
            write_video(output_path, frames.cpu(), fps=fps)
            
        except Exception as e:
            print(f"Error saving video: {e}")
            raise
