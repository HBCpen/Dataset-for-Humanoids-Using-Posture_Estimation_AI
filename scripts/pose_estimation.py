#!/usr/bin/env python3
"""
Pose Estimation Script
Uses MediaPipe to extract 3D skeleton data from videos or images.
"""

import os
import sys
import cv2
import json
import argparse
import yaml
import numpy as np
import mediapipe as mp
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class PoseEstimator:
    """MediaPipe-based pose estimator."""
    
    # Landmark names for MediaPipe Pose
    LANDMARK_NAMES = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]
    
    # Skeleton connections for visualization
    SKELETON_CONNECTIONS = [
        # Torso
        (11, 12), (11, 23), (12, 24), (23, 24),
        # Left arm
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        # Right arm
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
        # Left leg
        (23, 25), (25, 27), (27, 29), (27, 31),
        # Right leg
        (24, 26), (26, 28), (28, 30), (28, 32),
        # Face
        (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),
        (3, 7), (6, 8), (9, 10)
    ]
    
    def __init__(self, config: dict):
        """Initialize pose estimator with configuration."""
        self.config = config
        pose_config = config.get("pose_estimation", {})
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=pose_config.get("model_complexity", 2),
            smooth_landmarks=pose_config.get("smooth_landmarks", True),
            min_detection_confidence=pose_config.get("min_detection_confidence", 0.5),
            min_tracking_confidence=pose_config.get("min_tracking_confidence", 0.5),
            enable_segmentation=False
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.enable_3d = pose_config.get("enable_3d", True)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[Dict], np.ndarray]:
        """
        Process a single frame and extract pose landmarks.
        
        Args:
            frame: BGR image frame
            
        Returns:
            Tuple of (landmarks_dict, annotated_frame)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        landmarks_dict = None
        annotated_frame = frame.copy()
        
        if results.pose_landmarks:
            # Extract 2D landmarks (normalized)
            landmarks_2d = []
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks_2d.append({
                    "name": self.LANDMARK_NAMES[i],
                    "x": landmark.x,
                    "y": landmark.y,
                    "visibility": landmark.visibility
                })
            
            landmarks_dict = {
                "landmarks_2d": landmarks_2d,
                "timestamp": None  # Will be set by caller
            }
            
            # Extract 3D landmarks if available
            if self.enable_3d and results.pose_world_landmarks:
                landmarks_3d = []
                for i, landmark in enumerate(results.pose_world_landmarks.landmark):
                    landmarks_3d.append({
                        "name": self.LANDMARK_NAMES[i],
                        "x": landmark.x,  # meters, hip center origin
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    })
                landmarks_dict["landmarks_3d"] = landmarks_3d
            
            # Draw landmarks on frame
            vis_config = self.config.get("visualization", {})
            if vis_config.get("draw_skeleton", True):
                annotated_frame = self._draw_skeleton(
                    annotated_frame,
                    results.pose_landmarks,
                    vis_config
                )
        
        return landmarks_dict, annotated_frame
    
    def _draw_skeleton(
        self,
        frame: np.ndarray,
        landmarks,
        vis_config: dict
    ) -> np.ndarray:
        """Draw skeleton on frame."""
        h, w = frame.shape[:2]
        
        # Draw connections
        line_thickness = vis_config.get("line_thickness", 2)
        for connection in self.SKELETON_CONNECTIONS:
            start_idx, end_idx = connection
            start = landmarks.landmark[start_idx]
            end = landmarks.landmark[end_idx]
            
            if start.visibility > 0.5 and end.visibility > 0.5:
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                
                # Color gradient based on body part
                if start_idx < 11:  # Face
                    color = (255, 200, 100)
                elif start_idx < 17:  # Arms
                    color = (100, 255, 100)
                elif start_idx < 23:  # Hands
                    color = (100, 200, 255)
                else:  # Legs
                    color = (255, 100, 100)
                
                cv2.line(frame, start_point, end_point, color, line_thickness)
        
        # Draw keypoints
        keypoint_radius = vis_config.get("keypoint_radius", 4)
        for i, landmark in enumerate(landmarks.landmark):
            if landmark.visibility > 0.5:
                center = (int(landmark.x * w), int(landmark.y * h))
                cv2.circle(frame, center, keypoint_radius, (0, 255, 255), -1)
                cv2.circle(frame, center, keypoint_radius + 1, (0, 0, 0), 1)
        
        return frame
    
    def close(self):
        """Release resources."""
        self.pose.close()


def process_video(
    input_path: str,
    output_dir: str,
    config: dict,
    save_video: bool = True
) -> Dict:
    """
    Process a video file and extract pose data.
    
    Args:
        input_path: Path to input video
        output_dir: Directory for output files
        config: Configuration dictionary
        save_video: If True, save annotated video
        
    Returns:
        Dictionary with processing results and statistics
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    video_name = Path(input_path).stem
    
    # Initialize pose estimator
    estimator = PoseEstimator(config)
    
    # Initialize video writer if needed
    out = None
    if save_video:
        output_video = output_path / f"{video_name}_pose.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    # Process frames
    all_landmarks = []
    frames_with_pose = 0
    
    with tqdm(total=total_frames, desc=f"Processing {video_name}") as pbar:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            
            landmarks, annotated_frame = estimator.process_frame(frame)
            
            if landmarks:
                landmarks["frame_index"] = frame_idx
                landmarks["timestamp"] = timestamp
                all_landmarks.append(landmarks)
                frames_with_pose += 1
            
            if out:
                out.write(annotated_frame)
            
            frame_idx += 1
            pbar.update(1)
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    estimator.close()
    
    # Save pose data as JSON
    metadata = {
        "source_video": input_path,
        "video_properties": {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "duration_seconds": total_frames / fps if fps > 0 else 0
        },
        "processing_info": {
            "model": "MediaPipe Pose",
            "model_complexity": config.get("pose_estimation", {}).get("model_complexity", 2),
            "processed_at": datetime.now().isoformat(),
            "frames_with_pose": frames_with_pose,
            "detection_rate": frames_with_pose / total_frames if total_frames > 0 else 0
        },
        "landmark_names": PoseEstimator.LANDMARK_NAMES
    }
    
    output_data = {
        "metadata": metadata,
        "frames": all_landmarks
    }
    
    output_json = output_path / f"{video_name}_pose.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return {
        "input_path": input_path,
        "output_json": str(output_json),
        "output_video": str(output_path / f"{video_name}_pose.mp4") if save_video else None,
        "total_frames": total_frames,
        "frames_with_pose": frames_with_pose,
        "detection_rate": metadata["processing_info"]["detection_rate"]
    }


def process_image(
    input_path: str,
    output_dir: str,
    config: dict
) -> Dict:
    """Process a single image and extract pose data."""
    frame = cv2.imread(input_path)
    if frame is None:
        raise ValueError(f"Cannot read image: {input_path}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_name = Path(input_path).stem
    
    # Initialize pose estimator with static mode
    estimator = PoseEstimator(config)
    
    landmarks, annotated_frame = estimator.process_frame(frame)
    
    estimator.close()
    
    # Save annotated image
    output_image = output_path / f"{image_name}_pose.jpg"
    cv2.imwrite(str(output_image), annotated_frame)
    
    # Save pose data
    output_json = output_path / f"{image_name}_pose.json"
    
    if landmarks:
        landmarks["frame_index"] = 0
        landmarks["timestamp"] = 0.0
    
    output_data = {
        "metadata": {
            "source_image": input_path,
            "image_size": {"width": frame.shape[1], "height": frame.shape[0]},
            "model": "MediaPipe Pose",
            "processed_at": datetime.now().isoformat(),
            "pose_detected": landmarks is not None
        },
        "landmark_names": PoseEstimator.LANDMARK_NAMES,
        "landmarks": landmarks
    }
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return {
        "input_path": input_path,
        "output_json": str(output_json),
        "output_image": str(output_image),
        "pose_detected": landmarks is not None
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract pose data from videos or images using MediaPipe"
    )
    parser.add_argument(
        "input",
        help="Input video file, image file, or directory"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory",
        default=None
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to config file",
        default=None
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Don't save annotated video (JSON only)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    input_path = Path(args.input)
    
    # Determine output path
    if args.output:
        output_dir = Path(args.output)
    else:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "data" / "processed"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input is image or video
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    video_extensions = set(config["input"]["supported_formats"])
    
    if input_path.is_file():
        ext = input_path.suffix.lower()
        
        if ext in image_extensions:
            result = process_image(str(input_path), str(output_dir), config)
            print("\nImage processing complete!")
            print(f"  Input:  {result['input_path']}")
            print(f"  Output: {result['output_json']}")
            print(f"  Pose detected: {result['pose_detected']}")
        
        elif ext in video_extensions:
            result = process_video(
                str(input_path),
                str(output_dir),
                config,
                save_video=not args.no_video
            )
            print("\nVideo processing complete!")
            print(f"  Input:  {result['input_path']}")
            print(f"  Output JSON: {result['output_json']}")
            if result['output_video']:
                print(f"  Output Video: {result['output_video']}")
            print(f"  Detection rate: {result['detection_rate']:.1%}")
        else:
            print(f"Error: Unsupported file format: {ext}")
            sys.exit(1)
    
    elif input_path.is_dir():
        # Process all files in directory
        all_files = list(input_path.iterdir())
        videos = [f for f in all_files if f.suffix.lower() in video_extensions]
        images = [f for f in all_files if f.suffix.lower() in image_extensions]
        
        results = []
        
        for video in videos:
            try:
                result = process_video(
                    str(video),
                    str(output_dir),
                    config,
                    save_video=not args.no_video
                )
                result["status"] = "success"
                results.append(result)
                print(f"✓ {video.name}: {result['detection_rate']:.1%} detection rate")
            except Exception as e:
                results.append({"input_path": str(video), "status": "error", "error": str(e)})
                print(f"✗ {video.name}: {e}")
        
        for image in images:
            try:
                result = process_image(str(image), str(output_dir), config)
                result["status"] = "success"
                results.append(result)
                print(f"✓ {image.name}: pose {'detected' if result['pose_detected'] else 'not detected'}")
            except Exception as e:
                results.append({"input_path": str(image), "status": "error", "error": str(e)})
                print(f"✗ {image.name}: {e}")
        
        success = sum(1 for r in results if r.get("status") == "success")
        errors = sum(1 for r in results if r.get("status") == "error")
        
        print(f"\nProcessing complete!")
        print(f"  Successful: {success}")
        print(f"  Errors: {errors}")
    else:
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
