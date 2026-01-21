#!/usr/bin/env python3
"""
Main Pipeline Script - Run the complete pose estimation pipeline
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from preprocess import process_directory as preprocess_dir, preprocess_video
from pose_estimation import process_video as estimate_pose, process_image
from postprocess import postprocess_file
from export import export_file


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "default.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(input_path: str, output_dir: str = None, config: dict = None):
    """
    Run the complete pose estimation pipeline.
    
    Pipeline stages:
    1. Preprocess - Normalize video resolution
    2. Pose Estimation - Extract 3D skeleton data
    3. Postprocess - Filter, smooth, normalize data
    4. Export - Convert to various formats
    """
    input_path = Path(input_path)
    
    if config is None:
        config = load_config()
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "data"
    else:
        output_dir = Path(output_dir)
    
    # Create output directories
    interim_dir = output_dir / "interim"
    processed_dir = output_dir / "processed"
    export_dir = output_dir / "export"
    
    for d in [interim_dir, processed_dir, export_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Pose Estimation Pipeline for Humanoid Dataset Generation")
    print("=" * 60)
    
    # Check input type
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    video_ext = set(config["input"]["supported_formats"])
    
    if input_path.is_file():
        ext = input_path.suffix.lower()
        
        if ext in image_ext:
            print(f"\n[1/4] Processing image: {input_path.name}")
            result = process_image(str(input_path), str(processed_dir), config)
            
            print(f"\n[2/4] Postprocessing...")
            json_file = processed_dir / f"{input_path.stem}_pose.json"
            processed_file = processed_dir / f"{input_path.stem}_pose_processed.json"
            postprocess_file(str(json_file), str(processed_file), config)
            
            print(f"\n[3/4] Exporting...")
            export_file(str(processed_file), str(export_dir), config)
            
            print("\n" + "=" * 60)
            print("Pipeline completed successfully!")
            print(f"  Pose detected: {result.get('pose_detected', 'N/A')}")
            print(f"  Output: {export_dir}")
            
        elif ext in video_ext:
            video_name = input_path.stem
            
            print(f"\n[1/4] Preprocessing: {input_path.name}")
            preprocessed = interim_dir / f"{video_name}_processed.mp4"
            preprocess_video(str(input_path), str(preprocessed), config)
            
            print(f"\n[2/4] Pose Estimation...")
            result = estimate_pose(str(preprocessed), str(processed_dir), config)
            
            print(f"\n[3/4] Postprocessing...")
            json_file = processed_dir / f"{video_name}_processed_pose.json"
            processed_file = processed_dir / f"{video_name}_final.json"
            postprocess_file(str(json_file), str(processed_file), config)
            
            print(f"\n[4/4] Exporting...")
            export_file(str(processed_file), str(export_dir), config)
            
            print("\n" + "=" * 60)
            print("Pipeline completed successfully!")
            print(f"  Total frames: {result.get('total_frames', 'N/A')}")
            print(f"  Frames with pose: {result.get('frames_with_pose', 'N/A')}")
            print(f"  Detection rate: {result.get('detection_rate', 0):.1%}")
            print(f"  Output: {export_dir}")
        else:
            print(f"Error: Unsupported file format: {ext}")
            sys.exit(1)
    
    elif input_path.is_dir():
        print(f"\n[1/4] Preprocessing directory: {input_path}")
        preprocess_dir(str(input_path), str(interim_dir), config)
        
        print(f"\n[2/4] Running pose estimation on preprocessed videos...")
        for video in interim_dir.glob("*_processed.mp4"):
            print(f"  Processing: {video.name}")
            estimate_pose(str(video), str(processed_dir), config, save_video=True)
        
        print(f"\n[3/4] Postprocessing all pose data...")
        for json_file in processed_dir.glob("*_pose.json"):
            out_file = processed_dir / f"{json_file.stem}_final.json"
            postprocess_file(str(json_file), str(out_file), config)
        
        print(f"\n[4/4] Exporting to final formats...")
        for json_file in processed_dir.glob("*_final.json"):
            export_file(str(json_file), str(export_dir), config)
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print(f"  Output directory: {export_dir}")
    else:
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete pose estimation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py video.mp4                    # Process single video
  python pipeline.py image.jpg                    # Process single image
  python pipeline.py data/raw/                    # Process all files in directory
  python pipeline.py video.mp4 -o custom_output/  # Custom output directory
        """
    )
    parser.add_argument("input", help="Input video/image file or directory")
    parser.add_argument("-o", "--output", default=None, help="Output directory")
    parser.add_argument("-c", "--config", default=None, help="Config file path")
    
    args = parser.parse_args()
    
    config = load_config(args.config) if args.config else load_config()
    run_pipeline(args.input, args.output, config)


if __name__ == "__main__":
    main()
