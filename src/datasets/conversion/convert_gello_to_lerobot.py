#!/usr/bin/env python3
"""
Convert Gello PKL datasets to LeRobot v3.0 format.

Gello data structure (per PKL file = 1 timestep):
- wrist_rgb: (480, 640, 3) uint8
- wrist_depth: (480, 640, 1) uint16
- joint_positions: (8,) float - [j0-j6, gripper]
- joint_velocities: (8,) float
- ee_pos_quat: (7,) float - [x, y, z, qw, qx, qy, qz]
- gripper_position: float
- control: (8,) float - commanded joint positions

LeRobot v3 structure:
- meta/info.json: Central metadata (schema, fps, paths)
- meta/stats.json: Statistics for normalization
- meta/tasks.parquet: Task descriptions
- meta/episodes/chunk-XXX/file-XXX.parquet: Episode metadata
- data/chunk-XXX/file-XXX.parquet: Frame-by-frame data
- videos/<camera>/chunk-XXX/file-XXX.mp4: Video files
"""

import argparse
import os
import pickle
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import List

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from tqdm import tqdm


def load_episode_data(episode_dir: Path) -> List[dict]:
    """Load all PKL files from an episode directory in chronological order."""
    pkl_files = sorted(episode_dir.glob("*.pkl"))
    
    if not pkl_files:
        raise ValueError(f"No PKL files found in {episode_dir}")
    
    frames = []
    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            frames.append(data)
    
    return frames


def extract_task_name(episode_dir: Path) -> str:
    """Extract task name from episode directory structure.
    
    Expected structure: datasets/gello_data/gello/MMDD_HHMMSS/
    We use the timestamp as the episode identifier.
    """
    # Use parent directory name as task (e.g., "gello")
    # and episode directory name as episode ID
    task_name = episode_dir.parent.name
    return task_name


def convert_gello_to_lerobot(
    input_dir: str,
    output_dir: str,
    fps: int = 10,
    robot_type: str = "xarm",
    task_name: str = None,
    goal_description: str = None,
    overwrite: bool = False,
):
    """
    Convert Gello PKL episodes to LeRobot v3 format.
    
    Args:
        input_dir: Root directory containing episode folders
        output_dir: Output directory for LeRobot dataset
        fps: Frames per second of the dataset
        robot_type: Robot type identifier
        task_name: Optional task name (inferred from structure if not provided)
        goal_description: Optional goal description
        overwrite: If True, delete existing output directory
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Handle existing output directory
    if output_path.exists():
        if overwrite:
            print(f"[info] Removing existing output directory: {output_path}")
            # Use subprocess rm -rf for most reliable deletion
            try:
                result = subprocess.run(
                    ['rm', '-rf', str(output_path)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"[info] Directory removed via rm -rf")
            except subprocess.CalledProcessError as e:
                print(f"[warning] rm -rf failed: {e.stderr}")
                # Fallback to shutil
                def force_remove_readonly(func, path, exc_info):
                    """Error handler for shutil.rmtree to handle read-only files."""
                    os.chmod(path, 0o777)
                    func(path)
                shutil.rmtree(output_path, onerror=force_remove_readonly)
            
            # Wait for filesystem to sync
            time.sleep(1.5)
            
            # Verify deletion with detailed check
            for attempt in range(5):
                if not output_path.exists():
                    print(f"[info] Directory successfully deleted")
                    break
                
                # Check what's still there
                try:
                    contents = list(output_path.iterdir()) if output_path.exists() else []
                    print(f"[info] Directory still exists with {len(contents)} items (attempt {attempt + 1}/5)")
                    if contents:
                        print(f"[info] Sample contents: {[str(c.name) for c in contents[:3]]}")
                except Exception as e:
                    print(f"[info] Error checking directory: {e}")
                
                time.sleep(1.0)
            
            if output_path.exists():
                # Last resort: try to remove each file individually
                print(f"[warning] Attempting individual file removal...")
                try:
                    for root, dirs, files in os.walk(output_path, topdown=False):
                        for name in files:
                            file_path = os.path.join(root, name)
                            try:
                                os.chmod(file_path, 0o777)
                                os.remove(file_path)
                            except Exception as e:
                                print(f"[warning] Could not remove {file_path}: {e}")
                        for name in dirs:
                            dir_path = os.path.join(root, name)
                            try:
                                os.chmod(dir_path, 0o777)
                                os.rmdir(dir_path)
                            except Exception as e:
                                print(f"[warning] Could not remove {dir_path}: {e}")
                    os.rmdir(output_path)
                except Exception as e:
                    print(f"[error] Individual removal failed: {e}")
                
                time.sleep(1.0)
                
            if output_path.exists():
                raise RuntimeError(
                    f"Failed to remove directory after all attempts: {output_path}\n"
                    f"Please manually run: rm -rf {output_path}"
                )
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_path}\n"
                f"Use --overwrite flag to replace it, or choose a different output directory."
            )
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all episode directories (directories containing PKL files)
    episode_dirs = []
    for path in sorted(input_path.rglob("*")):
        if path.is_dir() and list(path.glob("*.pkl")):
            episode_dirs.append(path)
    
    if not episode_dirs:
        raise ValueError(f"No episode directories with PKL files found in {input_dir}")
    
    print(f"\n[info] Found {len(episode_dirs)} episodes")
    for ep_dir in episode_dirs[:5]:  # Show first 5
        print(f"  - {ep_dir.relative_to(input_path)}")
    if len(episode_dirs) > 5:
        print(f"  ... and {len(episode_dirs) - 5} more")
    
    # Infer task name if not provided
    if task_name is None:
        task_name = extract_task_name(episode_dirs[0])
    
    if goal_description is None:
        goal_description = f"Perform {task_name} task"
    
    # Define feature schema for LeRobot v3
    # State: 8 joint positions [j0-j6, gripper]
    # Action: 8 joint commands
    state_dim = 8
    action_dim = 8
    
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": {
                "motors": [
                    "joint_0",
                    "joint_1", 
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                    "joint_6",
                    "gripper",
                ]
            },
        },
        "observation.velocity": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": {
                "motors": [
                    "joint_0_vel",
                    "joint_1_vel",
                    "joint_2_vel", 
                    "joint_3_vel",
                    "joint_4_vel",
                    "joint_5_vel",
                    "joint_6_vel",
                    "gripper_vel",
                ]
            },
        },
        "observation.ee_pose": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": {
                "motors": [
                    "joint_0_cmd",
                    "joint_1_cmd",
                    "joint_2_cmd",
                    "joint_3_cmd",
                    "joint_4_cmd",
                    "joint_5_cmd",
                    "joint_6_cmd",
                    "gripper_cmd",
                ]
            },
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": fps,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "has_audio": False,
            },
        },
    }
    
    # Create dataset
    # Note: LeRobotDataset.create() will create the directory itself
    dataset = LeRobotDataset.create(
        repo_id=str(output_path),
        fps=fps,
        robot_type=robot_type,
        features=features,
    )
    
    # Process each episode
    for ep_idx, episode_dir in enumerate(tqdm(episode_dirs, desc="Converting episodes")):
        try:
            frames = load_episode_data(episode_dir)
        except Exception as e:
            print(f"\n Episode {ep_idx} ({episode_dir.name}): Failed to load - {e}")
            continue
        
        if len(frames) == 0:
            print(f"\n Episode {ep_idx} ({episode_dir.name}): No frames, skipping")
            continue
        
        # Convert each frame
        frame_count = 0
        for frame_idx, frame_data in enumerate(frames):
            try:
                # Extract data from frame
                joint_positions = np.asarray(
                    frame_data["joint_positions"], dtype=np.float32
                )
                joint_velocities = np.asarray(
                    frame_data["joint_velocities"], dtype=np.float32
                )
                ee_pos_quat = np.asarray(
                    frame_data["ee_pos_quat"], dtype=np.float32
                )
                control = np.asarray(
                    frame_data["control"], dtype=np.float32
                )
                wrist_rgb = frame_data["wrist_rgb"]
                
                # Ensure correct array sizes
                if len(joint_positions) != state_dim:
                    print(f"\n Frame {frame_idx}: Invalid joint_positions size {len(joint_positions)}, expected {state_dim}")
                    continue
                
                if len(control) != action_dim:
                    print(f"\n Frame {frame_idx}: Invalid control size {len(control)}, expected {action_dim}")
                    continue
                
                # Build frame dictionary
                frame_dict = {
                    "observation.state": joint_positions,
                    "observation.velocity": joint_velocities,
                    "observation.ee_pose": ee_pos_quat,
                    "action": control,
                    "task": goal_description,
                }
                
                # Add wrist camera image
                if wrist_rgb is not None and wrist_rgb.size > 0:
                    # Ensure uint8 format
                    if wrist_rgb.dtype != np.uint8:
                        wrist_rgb = (wrist_rgb * 255).clip(0, 255).astype(np.uint8)
                    
                    # Convert to PIL Image
                    frame_dict["observation.images.wrist"] = Image.fromarray(wrist_rgb)
                else:
                    # Black frame placeholder
                    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    frame_dict["observation.images.wrist"] = Image.fromarray(black_frame)
                
                # Add frame to dataset
                dataset.add_frame(frame_dict)
                frame_count += 1
                
            except Exception as e:
                print(f"\n Episode {ep_idx}, Frame {frame_idx}: Error - {e}")
                continue
        
        # Mark episode boundary (creates separate file-XXX for this episode)
        dataset.save_episode()
        print(f" Episode {ep_idx + 1}/{len(episode_dirs)} ({episode_dir.name}): {frame_count} frames -> file-{ep_idx:03d}")
    
    # Finalize dataset (encode videos, write metadata)
    dataset.finalize()
    print(f"\n[done] Converted {len(episode_dirs)} episodes -> {output_path}")
    print(f"[done] Structure:")
    print(f"  - meta/info.json, stats.json, tasks.parquet")
    print(f"  - data/chunk-000/file-000.parquet ... file-{len(episode_dirs)-1:03d}.parquet")
    print(f"  - videos/observation.images.wrist/chunk-000/file-000.mp4 ... file-{len(episode_dirs)-1:03d}.mp4")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gello PKL datasets to LeRobot v3.0 format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing episode folders with PKL files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./datasets/lerobot_gello",
        help="Output directory for LeRobot dataset",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second of the dataset (default: 10)",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="xarm",
        help="Robot type identifier (default: xarm)",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Task name (inferred from directory structure if not provided)",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Goal description for the task",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if it exists",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Gello PKL -> LeRobot v3.0 Converter")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"FPS: {args.fps}")
    print(f"Robot: {args.robot_type}")
    print(f"Task: {args.task_name or '(auto-detect)'}")
    print(f"Goal: {args.goal or '(auto-generate)'}")
    print("=" * 80)
    
    try:
        convert_gello_to_lerobot(
            input_dir=args.input,
            output_dir=args.output,
            fps=args.fps,
            robot_type=args.robot_type,
            task_name=args.task_name,
            goal_description=args.goal,
            overwrite=args.overwrite,
        )
        print("\n Conversion completed successfully!")
        return 0
    except Exception as e:
        print(f"\n Conversion failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
