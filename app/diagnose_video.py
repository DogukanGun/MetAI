#!/usr/bin/env python
"""
Video Diagnostics Tool
Checks video file properties, audio tracks, and formats
"""

import sys
import os

def diagnose_video(video_path):
    """Diagnose video file for audio issues."""
    print("=" * 70)
    print("VIDEO DIAGNOSTICS")
    print("=" * 70)
    print(f"\nVideo: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ ERROR: File not found: {video_path}")
        return
    
    file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
    print(f"File size: {file_size:.2f} MB")
    
    # Method 1: Try with moviepy
    print("\n" + "-" * 70)
    print("Method 1: Testing with moviepy...")
    try:
        # Try moviepy 2.x first
        try:
            from moviepy import VideoFileClip
        except ImportError:
            from moviepy.editor import VideoFileClip
        video = VideoFileClip(video_path)
        print(f"✓ Video loaded successfully")
        print(f"  Duration: {video.duration:.2f} seconds")
        print(f"  FPS: {video.fps}")
        print(f"  Size: {video.size}")
        
        if video.audio is None:
            print(f"❌ NO AUDIO TRACK FOUND")
        else:
            print(f"✓ Audio track exists")
            print(f"  Duration: {video.audio.duration:.2f} seconds")
            print(f"  FPS (sample rate): {video.audio.fps} Hz")
            
            # Try to extract a sample
            try:
                audio_array = video.audio.to_soundarray(fps=16000, nbytes=2)
                print(f"  ✓ Can extract audio: {len(audio_array)} samples")
            except Exception as e:
                print(f"  ❌ Cannot extract audio: {e}")
        
        video.close()
    except ImportError:
        print("⚠️  moviepy not available")
    except Exception as e:
        print(f"❌ moviepy failed: {e}")
    
    # Method 2: Try with librosa
    print("\n" + "-" * 70)
    print("Method 2: Testing with librosa...")
    try:
        import librosa
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, sr = librosa.load(video_path, sr=16000, duration=5.0)
        
        if len(audio) > 0:
            print(f"✓ Audio extracted successfully")
            print(f"  Samples: {len(audio)}")
            print(f"  Sample rate: {sr} Hz")
            print(f"  Duration: {len(audio)/sr:.2f} seconds")
            print(f"  Max amplitude: {abs(audio).max():.4f}")
            
            if abs(audio).max() < 0.001:
                print(f"  ⚠️  WARNING: Audio is very quiet (might be silent)")
        else:
            print(f"❌ NO AUDIO extracted")
    except ImportError:
        print("⚠️  librosa not available")
    except Exception as e:
        print(f"❌ librosa failed: {e}")
    
    # Method 3: Try with ffprobe (if available)
    print("\n" + "-" * 70)
    print("Method 3: Testing with ffprobe...")
    try:
        import subprocess
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 
             'stream=codec_type,codec_name', '-of', 'default=noprint_wrappers=1',
             video_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ ffprobe results:")
            print(result.stdout)
            
            if 'codec_type=audio' in result.stdout:
                print("✓ Audio stream detected by ffprobe")
            else:
                print("❌ No audio stream detected by ffprobe")
        else:
            print(f"⚠️  ffprobe failed: {result.stderr}")
    except FileNotFoundError:
        print("⚠️  ffprobe not available (install ffmpeg)")
    except Exception as e:
        print(f"❌ ffprobe error: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nIf all methods failed to find audio:")
    print("1. The video might not have an audio track")
    print("2. The audio codec might not be supported")
    print("3. Try converting: ffmpeg -i input.mp4 -c:v copy -c:a aac output.mp4")
    print("\nIf audio was found but extraction fails in the app:")
    print("1. Check file permissions")
    print("2. Ensure moviepy and librosa are installed correctly")
    print("3. Try updating: pip install --upgrade moviepy librosa")
    print("=" * 70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_video.py <video_file>")
        print("Example: python diagnose_video.py my_video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    diagnose_video(video_path)

