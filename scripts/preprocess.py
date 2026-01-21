#!/usr/bin/env python3
"""
Video Preprocessing Script
Handles video quality filtering, resolution normalization, and frame extraction.
"""

import os
import sys
import cv2
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_video_info(video_path: str) -> dict:
    """Extract video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    info = {
        "path": video_path,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
    }
    cap.release()
    return info


def calculate_new_size(
    original_width: int,
    original_height: int,
    max_width: int
) -> Tuple[int, int]:
    """Calculate new dimensions while maintaining aspect ratio."""
    if original_width <= max_width:
        return original_width, original_height
    
    scale = max_width / original_width
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Ensure even dimensions for video encoding
    new_width = new_width - (new_width % 2)
    new_height = new_height - (new_height % 2)
    
    return new_width, new_height


def preprocess_video(
    input_path: str,
    output_path: str,
    config: dict,
    extract_frames: bool = False
) -> dict:
    """
    Preprocess a single video file.
    
    Args:
        input_path: Path to input video
        output_path: Path for output video or frame directory
        config: Configuration dictionary
        extract_frames: If True, extract frames instead of writing video
    
    Returns:
        Dictionary with processing statistics
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate new dimensions
    max_width = config["input"]["max_width"]
    new_width, new_height = calculate_new_size(original_width, original_height, max_width)
    needs_resize = (new_width != original_width)
    
    frame_skip = config["input"]["frame_skip"]
    
    stats = {
        "input_path": input_path,
        "output_path": output_path,
        "original_resolution": f"{original_width}x{original_height}",
        "output_resolution": f"{new_width}x{new_height}",
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": 0,
        "resized": needs_resize
    }
    
    if extract_frames:
        # Create output directory for frames
        os.makedirs(output_path, exist_ok=True)
        
        frame_idx = 0
        processed_count = 0
        
        with tqdm(total=total_frames, desc=f"Extracting frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_skip == 0:
                    if needs_resize:
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    frame_path = os.path.join(output_path, f"frame_{processed_count:06d}.jpg")
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    processed_count += 1
                
                frame_idx += 1
                pbar.update(1)
        
        stats["processed_frames"] = processed_count
    else:
        # Write preprocessed video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = fps / frame_skip if frame_skip > 1 else fps
        out = cv2.VideoWriter(output_path, fourcc, out_fps, (new_width, new_height))
        
        if not out.isOpened():
            cap.release()
            raise ValueError(f"Cannot create output video: {output_path}")
        
        frame_idx = 0
        processed_count = 0
        
        with tqdm(total=total_frames, desc=f"Processing video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_skip == 0:
                    if needs_resize:
                        frame = cv2.resize(frame, (new_width, new_height))
                    out.write(frame)
                    processed_count += 1
                
                frame_idx += 1
                pbar.update(1)
        
        out.release()
        stats["processed_frames"] = processed_count
    
    cap.release()
    return stats


def process_directory(
    input_dir: str,
    output_dir: str,
    config: dict,
    extract_frames: bool = False
) -> list:
    """Process all videos in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    supported_formats = config["input"]["supported_formats"]
    videos = []
    
    for ext in supported_formats:
        videos.extend(input_path.glob(f"*{ext}"))
        videos.extend(input_path.glob(f"*{ext.upper()}"))
    
    results = []
    
    for video in videos:
        video_name = video.stem
        
        if extract_frames:
            out_path = output_path / video_name
        else:
            out_path = output_path / f"{video_name}_processed.mp4"
        
        try:
            stats = preprocess_video(
                str(video),
                str(out_path),
                config,
                extract_frames
            )
            stats["status"] = "success"
            results.append(stats)
            print(f"✓ Processed: {video.name}")
        except Exception as e:
            results.append({
                "input_path": str(video),
                "status": "error",
                "error": str(e)
            })
            print(f"✗ Error processing {video.name}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess videos for pose estimation pipeline"
    )
    parser.add_argument(
        "input",
        help="Input video file or directory"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path (file or directory)",
        default=None
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to config file",
        default=None
    )
    parser.add_argument(
        "-f", "--frames",
        action="store_true",
        help="Extract individual frames instead of video"
    )
    parser.add_argument(
        "-i", "--info",
        action="store_true",
        help="Only show video information, don't process"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    input_path = Path(args.input)
    
    if args.info:
        # Just show video info
        if input_path.is_file():
            info = get_video_info(str(input_path))
            print("\nVideo Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print("--info requires a video file path")
        return
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        project_root = Path(__file__).parent.parent
        output_path = project_root / "data" / "interim"
    
    if input_path.is_file():
        # Process single file
        if output_path.suffix == "":
            output_path.mkdir(parents=True, exist_ok=True)
            if args.frames:
                final_output = output_path / input_path.stem
            else:
                final_output = output_path / f"{input_path.stem}_processed.mp4"
        else:
            final_output = output_path
            final_output.parent.mkdir(parents=True, exist_ok=True)
        
        stats = preprocess_video(
            str(input_path),
            str(final_output),
            config,
            args.frames
        )
        
        print("\nProcessing complete!")
        print(f"  Input:  {stats['input_path']}")
        print(f"  Output: {stats['output_path']}")
        print(f"  Resolution: {stats['original_resolution']} → {stats['output_resolution']}")
        print(f"  Frames processed: {stats['processed_frames']}")
    
    elif input_path.is_dir():
        # Process directory
        results = process_directory(
            str(input_path),
            str(output_path),
            config,
            args.frames
        )
        
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
