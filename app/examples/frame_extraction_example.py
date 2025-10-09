"""
Frame Extraction Example

Demonstrates the LlamaIndex-style frame extraction approach.
Reference: https://www.llamaindex.ai/blog/multimodal-rag-for-advanced-video-processing-with-llamaindex-lancedb-33be4804822e

Usage:
    python frame_extraction_example.py path/to/video.mp4
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.stage1_input import VideoProcessor
import logging

logging.basicConfig(level=logging.INFO)


def extract_frames_example(video_path: str, output_folder: str = "./extracted_frames"):
    """
    Extract frames from video using LlamaIndex approach.
    
    Args:
        video_path: Path to video file
        output_folder: Directory to save extracted frames
    """
    print(f"\n{'='*60}")
    print("Frame Extraction Example (LlamaIndex Approach)")
    print(f"{'='*60}\n")
    
    # Initialize processor
    processor = VideoProcessor(video_path)
    
    # Get video info
    metadata = processor.get_video_metadata()
    print(f"Video Info:")
    print(f"  - Duration: {metadata.get('duration', 0):.2f} seconds")
    print(f"  - FPS: {metadata.get('fps', 0):.2f}")
    print(f"  - Resolution: {metadata.get('width', 0)}x{metadata.get('height', 0)}")
    print(f"  - Total Frames: {metadata.get('frame_count', 0)}")
    
    # Extract frames at 0.2 FPS (1 frame every 5 seconds)
    print(f"\nExtracting frames at 0.2 FPS (1 frame every 5 seconds)...")
    frame_paths = processor.extract_frames_to_files(
        output_folder=output_folder,
        fps=0.2
    )
    
    print(f"\n{'='*60}")
    print(f"Extraction Complete!")
    print(f"{'='*60}")
    print(f"  - Total frames extracted: {len(frame_paths)}")
    print(f"  - Output folder: {output_folder}")
    print(f"  - Frame naming: frame0001.png, frame0002.png, ...")
    
    # Show first few frame paths
    if len(frame_paths) > 0:
        print(f"\nFirst 5 frames:")
        for i, path in enumerate(frame_paths[:5]):
            timestamp = i * 5.0  # 5 seconds per frame
            print(f"  {i+1}. {os.path.basename(path)} (at {timestamp:.1f}s)")
    
    print(f"\n{'='*60}\n")
    
    return frame_paths


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python frame_extraction_example.py <video_path>")
        print("\nExample:")
        print("  python frame_extraction_example.py my_video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Extract frames
    frame_paths = extract_frames_example(video_path)
    
    print(f"Success! Extracted {len(frame_paths)} frames.")
    print(f"\nThese frames can be used for:")
    print("  - Multimodal RAG with LlamaIndex")
    print("  - Vector database indexing with LanceDB")
    print("  - Visual emotion analysis")
    print("  - Video summarization")
    print("  - Scene detection")
