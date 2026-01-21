#!/usr/bin/env python3
"""
Data Export Script - Convert pose data to various formats for robot learning
"""

import json
import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_csv(data: Dict, output_path: str, include_confidence: bool = True):
    """Export pose data to CSV format."""
    frames = data.get("frames", [])
    if not frames:
        return
    
    rows = []
    for frame in frames:
        row = {"frame_index": frame.get("frame_index"), "timestamp": frame.get("timestamp")}
        
        for key in ["landmarks_3d", "landmarks_2d"]:
            if key not in frame:
                continue
            for lm in frame[key]:
                prefix = f"{lm['name']}_{key[-2:]}"
                row[f"{prefix}_x"] = lm.get("x")
                row[f"{prefix}_y"] = lm.get("y")
                if "z" in lm:
                    row[f"{prefix}_z"] = lm.get("z")
                if include_confidence:
                    row[f"{prefix}_visibility"] = lm.get("visibility")
        
        if "joint_angles" in frame:
            for name, angle in frame["joint_angles"].items():
                row[f"angle_{name}"] = angle
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def to_npz(data: Dict, output_path: str):
    """Export pose data to NPZ format for fast loading."""
    frames = data.get("frames", [])
    if not frames:
        return
    
    n_frames = len(frames)
    
    if "landmarks_3d" in frames[0]:
        n_landmarks = len(frames[0]["landmarks_3d"])
        landmarks_3d = np.zeros((n_frames, n_landmarks, 3))
        visibility_3d = np.zeros((n_frames, n_landmarks))
        
        for i, frame in enumerate(frames):
            for j, lm in enumerate(frame["landmarks_3d"]):
                landmarks_3d[i, j] = [lm.get("x", 0), lm.get("y", 0), lm.get("z", 0)]
                visibility_3d[i, j] = lm.get("visibility", 0)
    else:
        landmarks_3d = None
        visibility_3d = None
    
    timestamps = np.array([f.get("timestamp", 0) for f in frames])
    frame_indices = np.array([f.get("frame_index", i) for i, f in enumerate(frames)])
    
    save_dict = {"timestamps": timestamps, "frame_indices": frame_indices}
    if landmarks_3d is not None:
        save_dict["landmarks_3d"] = landmarks_3d
        save_dict["visibility_3d"] = visibility_3d
    
    np.savez_compressed(output_path, **save_dict)


def to_humanoid_format(data: Dict, output_path: str, robot_type: str = "generic"):
    """Export to humanoid-specific format with joint angles and trajectories."""
    frames = data.get("frames", [])
    metadata = data.get("metadata", {})
    
    output = {
        "format_version": "1.0",
        "robot_type": robot_type,
        "source": metadata.get("source_video", "unknown"),
        "fps": metadata.get("video_properties", {}).get("fps", 30),
        "duration": metadata.get("video_properties", {}).get("duration_seconds", 0),
        "trajectory": []
    }
    
    for frame in frames:
        entry = {"t": frame.get("timestamp", 0), "joints": {}}
        
        if "joint_angles" in frame:
            entry["joints"] = frame["joint_angles"]
        
        if "landmarks_3d" in frame:
            lms = frame["landmarks_3d"]
            entry["pose"] = {
                "head": [lms[0]["x"], lms[0]["y"], lms[0]["z"]] if len(lms) > 0 else None,
                "left_hand": [lms[15]["x"], lms[15]["y"], lms[15]["z"]] if len(lms) > 15 else None,
                "right_hand": [lms[16]["x"], lms[16]["y"], lms[16]["z"]] if len(lms) > 16 else None,
                "left_foot": [lms[27]["x"], lms[27]["y"], lms[27]["z"]] if len(lms) > 27 else None,
                "right_foot": [lms[28]["x"], lms[28]["y"], lms[28]["z"]] if len(lms) > 28 else None,
                "hip_center": [(lms[23]["x"] + lms[24]["x"]) / 2,
                               (lms[23]["y"] + lms[24]["y"]) / 2,
                               (lms[23]["z"] + lms[24]["z"]) / 2] if len(lms) > 24 else None
            }
        
        output["trajectory"].append(entry)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def export_file(input_path: str, output_dir: str, config: dict) -> Dict:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    name = Path(input_path).stem.replace("_processed", "").replace("_pose", "")
    export_cfg = config.get("export", {})
    formats = export_cfg.get("formats", ["json", "csv"])
    
    exported = []
    
    if "json" in formats:
        out = output_path / f"{name}_humanoid.json"
        to_humanoid_format(data, str(out))
        exported.append(str(out))
    
    if "csv" in formats:
        out = output_path / f"{name}.csv"
        to_csv(data, str(out), export_cfg.get("include_confidence", True))
        exported.append(str(out))
    
    if "npz" in formats:
        out = output_path / f"{name}.npz"
        to_npz(data, str(out))
        exported.append(str(out))
    
    return {"input": input_path, "exported": exported}


def main():
    parser = argparse.ArgumentParser(description="Export pose data to various formats")
    parser.add_argument("input", help="Input JSON file or directory")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("-f", "--formats", nargs="+", default=None, help="Export formats: json, csv, npz")
    args = parser.parse_args()
    
    config = load_config(args.config)
    if args.formats:
        config.setdefault("export", {})["formats"] = args.formats
    
    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else Path(__file__).parent.parent / "data" / "export"
    
    if input_path.is_file():
        result = export_file(str(input_path), str(output_dir), config)
        print(f"Exported: {', '.join(result['exported'])}")
    elif input_path.is_dir():
        for f in list(input_path.glob("*_processed.json")) + list(input_path.glob("*_pose.json")):
            result = export_file(str(f), str(output_dir), config)
            print(f"✓ {f.name} -> {len(result['exported'])} files")


if __name__ == "__main__":
    main()
