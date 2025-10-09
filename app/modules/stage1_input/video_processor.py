"""Video Processing Module

Handles video ingestion and extraction of audio and visual components.
"""

import os
import numpy as np
import cv2
from typing import List, Tuple, Optional
import logging

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logging.warning("moviepy not available, using OpenCV for video processing")

import librosa


class VideoProcessor:
    """Processes video files to extract audio and visual components."""
    
    def __init__(self, video_path: str, output_dir: str = "data/processed"):
        """
        Initialize VideoProcessor.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save processed outputs
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
    
    def extract_audio(self, sample_rate: int = 16000) -> Tuple[np.ndarray, int, str]:
        """
        Extract audio from video.
        
        Args:
            sample_rate: Target audio sample rate in Hz
            
        Returns:
            Tuple of (audio_array, sample_rate, audio_path)
        """
        self.logger.info(f"Extracting audio from {self.video_path}")
        
        audio_path = os.path.join(self.output_dir, "audio.wav")
        
        if MOVIEPY_AVAILABLE:
            try:
                video = VideoFileClip(self.video_path)
                if video.audio is None:
                    self.logger.warning("No audio track found in video")
                    return np.array([]), sample_rate, ""
                
                video.audio.write_audiofile(
                    audio_path,
                    fps=sample_rate,
                    verbose=False,
                    logger=None
                )
                video.close()
            except Exception as e:
                self.logger.error(f"Error extracting audio with moviepy: {e}")
                # Fallback to librosa
                return self._extract_audio_with_librosa(sample_rate)
        else:
            return self._extract_audio_with_librosa(sample_rate)
        
        # Load with librosa for processing
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        
        self.logger.info(f"Audio extracted: {len(audio)} samples at {sr} Hz")
        return audio, sr, audio_path
    
    def _extract_audio_with_librosa(self, sample_rate: int) -> Tuple[np.ndarray, int, str]:
        """Fallback audio extraction using librosa."""
        audio_path = os.path.join(self.output_dir, "audio.wav")
        audio, sr = librosa.load(self.video_path, sr=sample_rate)
        
        # Save to file
        import soundfile as sf
        sf.write(audio_path, audio, sr)
        
        return audio, sr, audio_path
    
    def extract_frames(self, fps: int = 5) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract frames from video at specified FPS.
        
        Args:
            fps: Frames per second to extract
            
        Returns:
            Tuple of (frames_list, timestamps_list)
        """
        self.logger.info(f"Extracting frames at {fps} FPS")
        
        frames = []
        timestamps = []
        
        if MOVIEPY_AVAILABLE:
            try:
                video = VideoFileClip(self.video_path)
                duration = video.duration
                
                # Calculate frame extraction times
                frame_times = np.arange(0, duration, 1.0 / fps)
                
                for t in frame_times:
                    try:
                        frame = video.get_frame(t)
                        frames.append(frame)
                        timestamps.append(t)
                    except Exception as e:
                        self.logger.warning(f"Error extracting frame at {t}s: {e}")
                        continue
                
                video.close()
            except Exception as e:
                self.logger.error(f"Error with moviepy: {e}, falling back to OpenCV")
                return self._extract_frames_with_opencv(fps)
        else:
            return self._extract_frames_with_opencv(fps)
        
        self.logger.info(f"Extracted {len(frames)} frames")
        return frames, timestamps
    
    def _extract_frames_with_opencv(self, fps: int) -> Tuple[List[np.ndarray], List[float]]:
        """Fallback frame extraction using OpenCV."""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
        
        frames = []
        timestamps = []
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps) if fps < video_fps else 1
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                timestamps.append(frame_count / video_fps)
            
            frame_count += 1
        
        cap.release()
        
        self.logger.info(f"Extracted {len(frames)} frames using OpenCV")
        return frames, timestamps
    
    def extract_frames_to_files(self, output_folder: str, fps: float = 0.2) -> List[str]:
        """
        Extract frames from video and save as individual image files.
        Implementation based on LlamaIndex multimodal RAG approach.
        
        Reference: https://www.llamaindex.ai/blog/multimodal-rag-for-advanced-video-processing-with-llamaindex-lancedb-33be4804822e
        
        Args:
            output_folder: Directory to save extracted frames
            fps: Frames per second to extract (default 0.2 = 1 frame every 5 seconds)
            
        Returns:
            List of paths to saved frame images
        """
        self.logger.info(f"Extracting frames to files at {fps} FPS (1 frame every {1/fps:.1f} seconds)")
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        frame_paths = []
        
        if MOVIEPY_AVAILABLE:
            try:
                clip = VideoFileClip(self.video_path)
                
                # Extract frames at specified FPS
                frame_pattern = os.path.join(output_folder, "frame%04d.png")
                clip.write_images_sequence(frame_pattern, fps=fps)
                
                # Get list of created files
                duration = clip.duration
                num_frames = int(duration * fps)
                frame_paths = [
                    os.path.join(output_folder, f"frame{i+1:04d}.png")
                    for i in range(num_frames)
                    if os.path.exists(os.path.join(output_folder, f"frame{i+1:04d}.png"))
                ]
                
                clip.close()
                
                self.logger.info(f"Extracted {len(frame_paths)} frames to {output_folder}")
            except Exception as e:
                self.logger.error(f"Error extracting frames with moviepy: {e}")
                self.logger.info("Falling back to OpenCV method")
                return self._extract_frames_to_files_opencv(output_folder, fps)
        else:
            return self._extract_frames_to_files_opencv(output_folder, fps)
        
        return frame_paths
    
    def _extract_frames_to_files_opencv(self, output_folder: str, fps: float) -> List[str]:
        """Fallback frame-to-file extraction using OpenCV."""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
        
        frame_paths = []
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps) if fps < video_fps else 1
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Save frame as PNG
                frame_path = os.path.join(output_folder, f"frame{saved_count+1:04d}.png")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        
        self.logger.info(f"Extracted {len(frame_paths)} frames to {output_folder} using OpenCV")
        return frame_paths
    
    def get_video_metadata(self) -> dict:
        """
        Get metadata about the video.
        
        Returns:
            Dictionary with video metadata
        """
        metadata = {
            'path': self.video_path,
            'filename': os.path.basename(self.video_path)
        }
        
        try:
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                metadata['fps'] = cap.get(cv2.CAP_PROP_FPS)
                metadata['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                metadata['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                metadata['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                metadata['duration'] = metadata['frame_count'] / metadata['fps']
                cap.release()
        except Exception as e:
            self.logger.error(f"Error getting video metadata: {e}")
        
        return metadata
    
    def process_video(self, fps: int = 5, sample_rate: int = 16000) -> dict:
        """
        Complete video processing pipeline.
        
        Args:
            fps: Frames per second for frame extraction
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary with processed data
        """
        self.logger.info("Starting video processing pipeline")
        
        # Get metadata
        metadata = self.get_video_metadata()
        
        # Extract audio
        audio, sr, audio_path = self.extract_audio(sample_rate)
        
        # Extract frames
        frames, timestamps = self.extract_frames(fps)
        
        result = {
            'metadata': metadata,
            'audio': audio,
            'sample_rate': sr,
            'audio_path': audio_path,
            'frames': frames,
            'frame_timestamps': timestamps
        }
        
        self.logger.info("Video processing complete")
        return result
