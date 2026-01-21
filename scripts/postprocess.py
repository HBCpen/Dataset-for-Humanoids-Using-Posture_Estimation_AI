#!/usr/bin/env python3
"""
Pose Data Postprocessing Script
Normalizes, filters, and refines extracted pose data.
"""

import os
import sys
import json
import argparse
import yaml
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from typing import Dict, List


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_coordinates(landmarks: List[Dict], mode: str = "hip_center") -> List[Dict]:
    if not landmarks:
        return landmarks
    coords = np.array([[l["x"], l["y"], l["z"]] for l in landmarks])
    
    if mode == "hip_center":
        hip_center = (coords[23] + coords[24]) / 2
        coords = coords - hip_center
    elif mode == "bounding_box":
        min_c = coords.min(axis=0)
        max_c = coords.max(axis=0)
        range_c = np.where((max_c - min_c) == 0, 1, max_c - min_c)
        coords = (coords - (max_c + min_c) / 2) / (range_c / 2)
    
    return [{**l, "x": float(coords[i, 0]), "y": float(coords[i, 1]), "z": float(coords[i, 2])} for i, l in enumerate(landmarks)]


def apply_temporal_smoothing(frames: List[Dict], sigma: float = 1.0, use_3d: bool = True) -> List[Dict]:
    if len(frames) < 3:
        return frames
    key = "landmarks_3d" if use_3d else "landmarks_2d"
    if not all(key in f for f in frames):
        return frames
    
    n_landmarks = len(frames[0][key])
    n_dims = 3 if use_3d else 2
    traj = np.array([[[l["x"], l["y"]] + ([l["z"]] if use_3d else []) for l in f[key]] for f in frames])
    
    smoothed = np.zeros_like(traj)
    for j in range(n_landmarks):
        for d in range(n_dims):
            smoothed[:, j, d] = gaussian_filter1d(traj[:, j, d], sigma=sigma)
    
    result = []
    for i, f in enumerate(frames):
        new_f = {**f, key: [{**l, "x": float(smoothed[i, j, 0]), "y": float(smoothed[i, j, 1])} | ({"z": float(smoothed[i, j, 2])} if use_3d else {}) for j, l in enumerate(f[key])]}
        result.append(new_f)
    return result


def compute_joint_angles(landmarks: List[Dict]) -> Dict[str, float]:
    def angle(v1, v2):
        n1, n2 = v1 / (np.linalg.norm(v1) + 1e-6), v2 / (np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1, 1)))
    
    def get(idx):
        if idx < len(landmarks) and landmarks[idx].get("x"):
            return np.array([landmarks[idx]["x"], landmarks[idx]["y"], landmarks[idx]["z"]])
        return None
    
    angles = {}
    for side, (sh, el, wr) in [("left", (11, 13, 15)), ("right", (12, 14, 16))]:
        if all(get(i) is not None for i in [sh, el, wr]):
            angles[f"{side}_elbow"] = angle(get(sh) - get(el), get(wr) - get(el))
    for side, (hp, kn, an) in [("left", (23, 25, 27)), ("right", (24, 26, 28))]:
        if all(get(i) is not None for i in [hp, kn, an]):
            angles[f"{side}_knee"] = angle(get(hp) - get(kn), get(an) - get(kn))
    return angles


def postprocess_file(input_path: str, output_path: str, config: dict) -> Dict:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    frames = data.get("frames", [])
    post_cfg = config.get("postprocess", {})
    use_3d = config.get("pose_estimation", {}).get("enable_3d", True)
    
    if post_cfg.get("temporal_smoothing", False):
        frames = apply_temporal_smoothing(frames, sigma=post_cfg.get("smoothing_window", 5) / 3.0, use_3d=use_3d)
    
    if post_cfg.get("normalize", False) and use_3d:
        for f in frames:
            if "landmarks_3d" in f:
                f["landmarks_3d"] = normalize_coordinates(f["landmarks_3d"])
    
    for f in frames:
        if "landmarks_3d" in f:
            f["joint_angles"] = compute_joint_angles(f["landmarks_3d"])
    
    data["frames"] = frames
    data.setdefault("metadata", {})["postprocessing"] = post_cfg
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return {"input_path": input_path, "output_path": output_path, "frames_processed": len(frames)}


def main():
    parser = argparse.ArgumentParser(description="Postprocess pose data")
    parser.add_argument("input", help="Input JSON file or directory")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("-c", "--config", default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    input_path = Path(args.input)
    
    if input_path.is_file():
        output_path = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_processed.json"
        result = postprocess_file(str(input_path), str(output_path), config)
        print(f"Processed: {result['output_path']}")
    elif input_path.is_dir():
        output_dir = Path(args.output) if args.output else input_path / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        for f in input_path.glob("*_pose.json"):
            postprocess_file(str(f), str(output_dir / f"{f.stem}_processed.json"), config)
            print(f"✓ {f.name}")


if __name__ == "__main__":
    main()
