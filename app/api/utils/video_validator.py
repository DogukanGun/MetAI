"""
Video Validator

Validates uploaded video files.
"""

import os
import cv2
from fastapi import UploadFile, HTTPException
from typing import List


class VideoValidator:
    """Validator for video file uploads."""
    
    ALLOWED_EXTENSIONS = ['.mp4', '.avi', '.mov', '.webm', '.mkv', '.flv']
    ALLOWED_MIME_TYPES = [
        'video/mp4',
        'video/x-msvideo',  # AVI
        'video/quicktime',  # MOV
        'video/webm',
        'video/x-matroska',  # MKV
        'video/x-flv'
    ]
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB
    MIN_DURATION = 1.0  # 1 second
    MAX_DURATION = 600.0  # 10 minutes
    
    @classmethod
    def validate_upload(cls, video: UploadFile) -> None:
        """
        Validate uploaded video file.
        
        Args:
            video: Uploaded file
            
        Raises:
            HTTPException: If validation fails
        """
        # Check filename
        if not video.filename:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": "ValidationError",
                    "message": "No filename provided",
                    "details": {}
                }
            )
        
        # Check file extension
        file_ext = os.path.splitext(video.filename)[1].lower()
        if file_ext not in cls.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": "ValidationError",
                    "message": f"Invalid file extension: {file_ext}",
                    "details": {
                        "allowed_extensions": cls.ALLOWED_EXTENSIONS,
                        "received_extension": file_ext
                    }
                }
            )
        
        # Check MIME type
        if video.content_type and video.content_type not in cls.ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": "ValidationError",
                    "message": f"Invalid MIME type: {video.content_type}",
                    "details": {
                        "allowed_mime_types": cls.ALLOWED_MIME_TYPES,
                        "received_mime_type": video.content_type
                    }
                }
            )
    
    @classmethod
    def validate_video_file(cls, video_path: str) -> None:
        """
        Validate video file content.
        
        Args:
            video_path: Path to video file
            
        Raises:
            HTTPException: If validation fails
        """
        # Check file exists
        if not os.path.exists(video_path):
            raise HTTPException(
                status_code=404,
                detail={
                    "success": False,
                    "error": "ValidationError",
                    "message": "Video file not found",
                    "details": {"path": video_path}
                }
            )
        
        # Check file size
        file_size = os.path.getsize(video_path)
        if file_size > cls.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": "ValidationError",
                    "message": f"File size exceeds maximum allowed size",
                    "details": {
                        "max_size_mb": cls.MAX_FILE_SIZE / (1024 * 1024),
                        "file_size_mb": file_size / (1024 * 1024)
                    }
                }
            )
        
        # Validate video content with OpenCV
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise HTTPException(
                    status_code=400,
                    detail={
                        "success": False,
                        "error": "ValidationError",
                        "message": "Unable to open video file",
                        "details": {"path": video_path}
                    }
                )
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps <= 0 or frame_count <= 0:
                cap.release()
                raise HTTPException(
                    status_code=400,
                    detail={
                        "success": False,
                        "error": "ValidationError",
                        "message": "Invalid video properties",
                        "details": {
                            "fps": fps,
                            "frame_count": frame_count
                        }
                    }
                )
            
            duration = frame_count / fps
            
            # Check duration
            if duration < cls.MIN_DURATION:
                cap.release()
                raise HTTPException(
                    status_code=400,
                    detail={
                        "success": False,
                        "error": "ValidationError",
                        "message": f"Video too short (minimum {cls.MIN_DURATION}s)",
                        "details": {
                            "min_duration": cls.MIN_DURATION,
                            "video_duration": duration
                        }
                    }
                )
            
            if duration > cls.MAX_DURATION:
                cap.release()
                raise HTTPException(
                    status_code=400,
                    detail={
                        "success": False,
                        "error": "ValidationError",
                        "message": f"Video too long (maximum {cls.MAX_DURATION}s)",
                        "details": {
                            "max_duration": cls.MAX_DURATION,
                            "video_duration": duration
                        }
                    }
                )
            
            cap.release()
        
        except HTTPException:
            raise
        
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": "ValidationError",
                    "message": f"Error validating video: {str(e)}",
                    "details": {}
                }
            )
