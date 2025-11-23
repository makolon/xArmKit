#!/usr/bin/env python3
"""
Convert Gello PKL datasets to HDF5 format with optional annotations.

This script converts Gello PKL episode data to HDF5 format and optionally
adds annotations including:
- Bounding boxes (via GroundingDINO)
- Segmentation masks (via SAM2)
- 6D poses (via FoundationPose)

Gello data structure (per PKL file = 1 timestep):
- wrist_rgb: (480, 640, 3) uint8
- wrist_depth: (480, 640, 1) uint16
- joint_positions: (8,) float - [j0-j6, gripper]
- joint_velocities: (8,) float
- ee_pos_quat: (7,) float - [x, y, z, qw, qx, qy, qz]
- gripper_position: float
- control: (8,) float - commanded joint positions

HDF5 output structure:
episode_0/
  ├─ observations/
  │   ├─ wrist_rgb (T, H, W, 3)
  │   ├─ wrist_depth (T, H, W, 1)
  │   ├─ joint_positions (T, 8)
  │   ├─ joint_velocities (T, 8)
  │   ├─ ee_pos_quat (T, 7)
  │   └─ gripper_position (T,)
  ├─ actions/
  │   └─ control (T, 8)
  └─ annotations/ (if enabled)
      ├─ bounding_boxes/
      │   └─ <object_name> (T, 4) [x1, y1, x2, y2]
      ├─ segmentation_masks/
      │   └─ <object_name> (T, H, W)
      └─ poses/
          └─ <object_name> (T, 7) [x, y, z, qw, qx, qy, qz]
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.groundingdino_utils import GroundingDINODetector
from utils.sam_utils import SAM2Segmenter
from utils.foundationpose_utils import FoundationPoseEstimator


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


def create_camera_intrinsic(
    width: int = 640,
    height: int = 480,
    fx: float = 605.0,
    fy: float = 605.0,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> np.ndarray:
    """
    Create camera intrinsic matrix.
    
    Args:
        width: Image width
        height: Image height
        fx: Focal length in x
        fy: Focal length in y
        cx: Principal point x (default: width/2)
        cy: Principal point y (default: height/2)
    
    Returns:
        Camera intrinsic matrix K (3, 3)
    """
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return K


def annotate_episode(
    frames: List[dict],
    object_names: List[str],
    mesh_paths: Optional[Dict[str, str]] = None,
    detect_bbox: bool = True,
    detect_mask: bool = True,
    estimate_pose: bool = True,
    grounding_dino_detector: Optional[GroundingDINODetector] = None,
    sam2_segmenter: Optional[SAM2Segmenter] = None,
    camera_intrinsic: Optional[np.ndarray] = None,
    foundationpose_score_ckpt: Optional[str] = None,
    foundationpose_refine_ckpt: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Annotate episode with bounding boxes, masks, and poses.
    
    Args:
        frames: List of frame dictionaries from PKL files
        object_names: List of object names to detect
        mesh_paths: Dict mapping object names to mesh file paths (for pose estimation)
        detect_bbox: Whether to detect bounding boxes
        detect_mask: Whether to detect segmentation masks
        estimate_pose: Whether to estimate 6D poses
        grounding_dino_detector: Initialized GroundingDINO detector
        sam2_segmenter: Initialized SAM2 segmenter
        camera_intrinsic: Camera intrinsic matrix (3, 3)
        foundationpose_score_ckpt: Path to FoundationPose score checkpoint
        foundationpose_refine_ckpt: Path to FoundationPose refine checkpoint
        device: Device to run on
    
    Returns:
        Dictionary containing annotations:
        {
            'bounding_boxes': {obj_name: (T, 4)},
            'segmentation_masks': {obj_name: (T, H, W)},
            'poses': {obj_name: (T, 7)}
        }
    """
    T = len(frames)
    H, W = frames[0]["wrist_rgb"].shape[:2]
    
    annotations = {
        "bounding_boxes": {},
        "segmentation_masks": {},
        "poses": {},
    }
    
    # Initialize storage for each object
    if detect_bbox:
        for obj_name in object_names:
            annotations["bounding_boxes"][obj_name] = np.zeros((T, 4), dtype=np.float32)
    
    if detect_mask:
        for obj_name in object_names:
            annotations["segmentation_masks"][obj_name] = np.zeros((T, H, W), dtype=np.uint8)
    
    if estimate_pose:
        for obj_name in object_names:
            annotations["poses"][obj_name] = np.zeros((T, 7), dtype=np.float32)
        
        # Initialize FoundationPose estimators
        if mesh_paths is None:
            raise ValueError("mesh_paths must be provided for pose estimation")
        
        pose_estimators = {}
        for obj_name in object_names:
            if obj_name not in mesh_paths:
                print(f"[Warning] No mesh path for '{obj_name}', skipping pose estimation")
                continue
            
            print(f"[Info] Initializing FoundationPose for '{obj_name}'")
            pose_estimators[obj_name] = FoundationPoseEstimator(
                mesh_path=mesh_paths[obj_name],
                score_checkpoint=foundationpose_score_ckpt,
                refine_checkpoint=foundationpose_refine_ckpt,
                device=device,
                debug=0,
            )
    
    # Process each frame
    print(f"\n[Info] Annotating {T} frames...")
    
    for frame_idx, frame_data in enumerate(tqdm(frames, desc="Annotating")):
        rgb = frame_data["wrist_rgb"]  # (H, W, 3)
        depth = frame_data["wrist_depth"].squeeze()  # (H, W)
        
        # Convert depth from uint16 to meters (assuming mm storage)
        depth_m = depth.astype(np.float32) / 1000.0
        
        # Step 1: Detect bounding boxes
        if detect_bbox and grounding_dino_detector is not None:
            # Create text prompt from object names
            text_prompt = " . ".join(object_names)
            
            # Detect objects
            boxes, phrases, scores = grounding_dino_detector.detect(
                rgb, text_prompt, with_logits=False
            )
            
            # Convert boxes to pixel coordinates
            boxes_xyxy = grounding_dino_detector.boxes_to_xyxy(boxes, (H, W))
            
            # Assign boxes to objects
            for box, phrase in zip(boxes_xyxy, phrases):
                # Find matching object name
                for obj_name in object_names:
                    if obj_name.lower() in phrase.lower():
                        annotations["bounding_boxes"][obj_name][frame_idx] = box.cpu().numpy()
                        break
        
        # Step 2: Segment objects
        if detect_mask and sam2_segmenter is not None:
            sam2_segmenter.set_image(rgb)
            
            for obj_name in object_names:
                bbox = annotations["bounding_boxes"].get(obj_name)
                if bbox is not None and not np.all(bbox[frame_idx] == 0):
                    box = bbox[frame_idx]
                    masks, scores, _ = sam2_segmenter.segment_from_box(
                        box, multimask_output=False
                    )
                    
                    if len(masks) > 0:
                        annotations["segmentation_masks"][obj_name][frame_idx] = masks[0].astype(np.uint8)
        
        # Step 3: Estimate poses
        if estimate_pose and camera_intrinsic is not None:
            for obj_name in object_names:
                if obj_name not in pose_estimators:
                    continue
                
                mask = annotations["segmentation_masks"].get(obj_name)
                if mask is not None and mask[frame_idx].sum() > 0:
                    estimator = pose_estimators[obj_name]
                    
                    # Register on first frame, track on subsequent frames
                    if frame_idx == 0 or not estimator.is_tracking:
                        pose_matrix = estimator.register(
                            rgb, depth_m, mask[frame_idx], camera_intrinsic
                        )
                    else:
                        try:
                            pose_matrix = estimator.track(
                                rgb, depth_m, camera_intrinsic
                            )
                        except:
                            # Re-register if tracking fails
                            pose_matrix = estimator.register(
                                rgb, depth_m, mask[frame_idx], camera_intrinsic
                            )
                    
                    # Convert to position + quaternion
                    pose_vec = FoundationPoseEstimator.pose_to_position_quaternion(pose_matrix)
                    annotations["poses"][obj_name][frame_idx] = pose_vec
    
    return annotations


def convert_gello_to_hdf(
    input_dir: str,
    output_file: str,
    objects: Optional[List[str]] = None,
    mesh_dir: Optional[str] = None,
    detect_bbox: bool = False,
    detect_mask: bool = False,
    estimate_pose: bool = False,
    grounding_dino_config: Optional[str] = None,
    grounding_dino_checkpoint: Optional[str] = None,
    sam2_checkpoint: Optional[str] = None,
    sam2_config: Optional[str] = None,
    foundationpose_score_ckpt: Optional[str] = None,
    foundationpose_refine_ckpt: Optional[str] = None,
    camera_fx: float = 605.0,
    camera_fy: float = 605.0,
    device: str = "cuda",
    overwrite: bool = False,
):
    """
    Convert Gello PKL episodes to HDF5 format with optional annotations.
    
    Args:
        input_dir: Root directory containing episode folders
        output_file: Output HDF5 file path
        objects: List of object names to detect/track
        mesh_dir: Directory containing object mesh files (.obj, .ply)
        detect_bbox: Whether to detect bounding boxes
        detect_mask: Whether to detect segmentation masks
        estimate_pose: Whether to estimate 6D poses
        grounding_dino_config: Path to GroundingDINO config
        grounding_dino_checkpoint: Path to GroundingDINO checkpoint
        sam2_checkpoint: Path to SAM2 checkpoint
        sam2_config: Path to SAM2 config
        foundationpose_score_ckpt: Path to FoundationPose score checkpoint
        foundationpose_refine_ckpt: Path to FoundationPose refine checkpoint
        camera_fx: Camera focal length x
        camera_fy: Camera focal length y
        device: Device to run on
        overwrite: Whether to overwrite existing output file
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    # Check if output exists
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}\n"
            f"Use --overwrite flag to replace it."
        )
    
    # Find all episode directories
    episode_dirs = []
    for path in sorted(input_path.rglob("*")):
        if path.is_dir() and list(path.glob("*.pkl")):
            episode_dirs.append(path)
    
    if not episode_dirs:
        raise ValueError(f"No episode directories with PKL files found in {input_dir}")
    
    print(f"\n[Info] Found {len(episode_dirs)} episodes")
    for ep_dir in episode_dirs[:5]:
        print(f"  - {ep_dir.relative_to(input_path)}")
    if len(episode_dirs) > 5:
        print(f"  ... and {len(episode_dirs) - 5} more")
    
    # Initialize annotators if needed
    grounding_dino_detector = None
    sam2_segmenter = None
    mesh_paths = None
    
    if detect_bbox or detect_mask or estimate_pose:
        if objects is None or len(objects) == 0:
            raise ValueError("--objects must be specified for annotation")
        
        print(f"\n[Info] Target objects: {', '.join(objects)}")
    
    if detect_bbox:
        if grounding_dino_config is None or grounding_dino_checkpoint is None:
            raise ValueError("GroundingDINO config and checkpoint required for bbox detection")
        
        print("\n[Info] Initializing GroundingDINO...")
        grounding_dino_detector = GroundingDINODetector(
            config_path=grounding_dino_config,
            checkpoint_path=grounding_dino_checkpoint,
            device=device,
        )
    
    if detect_mask:
        if sam2_checkpoint is None:
            raise ValueError("SAM2 checkpoint required for mask detection")
        
        print("\n[Info] Initializing SAM2...")
        sam2_segmenter = SAM2Segmenter(
            checkpoint_path=sam2_checkpoint,
            model_cfg=sam2_config or "configs/sam2.1/sam2.1_hiera_l.yaml",
            device=device,
        )
    
    if estimate_pose:
        if mesh_dir is None:
            raise ValueError("--mesh-dir required for pose estimation")
        
        mesh_dir_path = Path(mesh_dir)
        mesh_paths = {}
        
        for obj_name in objects:
            # Look for mesh file
            mesh_file = None
            for ext in [".obj", ".ply", ".stl"]:
                candidate = mesh_dir_path / f"{obj_name}{ext}"
                if candidate.exists():
                    mesh_file = str(candidate)
                    break
            
            if mesh_file is None:
                print(f"[Warning] No mesh file found for '{obj_name}' in {mesh_dir_path}")
            else:
                mesh_paths[obj_name] = mesh_file
                print(f"[Info] Found mesh for '{obj_name}': {mesh_file}")
    
    # Create camera intrinsic
    camera_intrinsic = None
    if estimate_pose:
        camera_intrinsic = create_camera_intrinsic(fx=camera_fx, fy=camera_fy)
        print(f"\n[Info] Camera intrinsic:\n{camera_intrinsic}")
    
    # Create output HDF5 file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, "w") as hf:
        print(f"\n[Info] Creating HDF5 file: {output_path}")
        
        # Process each episode
        for ep_idx, episode_dir in enumerate(episode_dirs):
            print(f"\n{'='*80}")
            print(f"Processing episode {ep_idx + 1}/{len(episode_dirs)}: {episode_dir.name}")
            print(f"{'='*80}")
            
            # Load episode data
            frames = load_episode_data(episode_dir)
            T = len(frames)
            
            print(f"[Info] Loaded {T} frames")
            
            # Create episode group
            ep_group = hf.create_group(f"episode_{ep_idx}")
            
            # Extract observation and action data
            obs_group = ep_group.create_group("observations")
            action_group = ep_group.create_group("actions")
            
            # Stack data across time
            wrist_rgb = np.stack([f["wrist_rgb"] for f in frames])  # (T, H, W, 3)
            wrist_depth = np.stack([f["wrist_depth"] for f in frames])  # (T, H, W, 1)
            joint_positions = np.stack([f["joint_positions"] for f in frames])  # (T, 8)
            joint_velocities = np.stack([f["joint_velocities"] for f in frames])  # (T, 8)
            ee_pos_quat = np.stack([f["ee_pos_quat"] for f in frames])  # (T, 7)
            gripper_position = np.array([f["gripper_position"] for f in frames])  # (T,)
            control = np.stack([f["control"] for f in frames])  # (T, 8)
            
            # Save observations
            obs_group.create_dataset("wrist_rgb", data=wrist_rgb, compression="gzip")
            obs_group.create_dataset("wrist_depth", data=wrist_depth, compression="gzip")
            obs_group.create_dataset("joint_positions", data=joint_positions)
            obs_group.create_dataset("joint_velocities", data=joint_velocities)
            obs_group.create_dataset("ee_pos_quat", data=ee_pos_quat)
            obs_group.create_dataset("gripper_position", data=gripper_position)
            
            # Save actions
            action_group.create_dataset("control", data=control)
            
            print(f"[Info] Saved observations and actions")
            
            # Annotate if requested
            if detect_bbox or detect_mask or estimate_pose:
                annotations = annotate_episode(
                    frames=frames,
                    object_names=objects,
                    mesh_paths=mesh_paths,
                    detect_bbox=detect_bbox,
                    detect_mask=detect_mask,
                    estimate_pose=estimate_pose,
                    grounding_dino_detector=grounding_dino_detector,
                    sam2_segmenter=sam2_segmenter,
                    camera_intrinsic=camera_intrinsic,
                    foundationpose_score_ckpt=foundationpose_score_ckpt,
                    foundationpose_refine_ckpt=foundationpose_refine_ckpt,
                    device=device,
                )
                
                # Save annotations
                annot_group = ep_group.create_group("annotations")
                
                if detect_bbox:
                    bbox_group = annot_group.create_group("bounding_boxes")
                    for obj_name, bboxes in annotations["bounding_boxes"].items():
                        bbox_group.create_dataset(obj_name, data=bboxes)
                    print(f"[Info] Saved bounding boxes")
                
                if detect_mask:
                    mask_group = annot_group.create_group("segmentation_masks")
                    for obj_name, masks in annotations["segmentation_masks"].items():
                        mask_group.create_dataset(obj_name, data=masks, compression="gzip")
                    print(f"[Info] Saved segmentation masks")
                
                if estimate_pose:
                    pose_group = annot_group.create_group("poses")
                    for obj_name, poses in annotations["poses"].items():
                        pose_group.create_dataset(obj_name, data=poses)
                    print(f"[Info] Saved poses")
    
    print(f"\n{'='*80}")
    print(f"[Done] Converted {len(episode_dirs)} episodes to {output_path}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gello PKL datasets to HDF5 format with optional annotations"
    )
    
    # Input/Output
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing episode folders with PKL files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )
    
    # Annotation options
    parser.add_argument(
        "--objects",
        type=str,
        nargs="+",
        default=None,
        help="Object names to detect/track (e.g., 'cup' 'banana' 'plate')",
    )
    parser.add_argument(
        "--detect-bbox",
        action="store_true",
        help="Detect bounding boxes using GroundingDINO",
    )
    parser.add_argument(
        "--detect-mask",
        action="store_true",
        help="Detect segmentation masks using SAM2",
    )
    parser.add_argument(
        "--estimate-pose",
        action="store_true",
        help="Estimate 6D poses using FoundationPose",
    )
    
    # Model paths
    parser.add_argument(
        "--grounding-dino-config",
        type=str,
        default=None,
        help="Path to GroundingDINO config file",
    )
    parser.add_argument(
        "--grounding-dino-checkpoint",
        type=str,
        default=None,
        help="Path to GroundingDINO checkpoint",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        type=str,
        default=None,
        help="Path to SAM2 checkpoint",
    )
    parser.add_argument(
        "--sam2-config",
        type=str,
        default=None,
        help="Path to SAM2 config file",
    )
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default=None,
        help="Directory containing object mesh files",
    )
    parser.add_argument(
        "--foundationpose-score-ckpt",
        type=str,
        default=None,
        help="Path to FoundationPose score predictor checkpoint",
    )
    parser.add_argument(
        "--foundationpose-refine-ckpt",
        type=str,
        default=None,
        help="Path to FoundationPose refine predictor checkpoint",
    )
    
    # Camera parameters
    parser.add_argument(
        "--camera-fx",
        type=float,
        default=605.0,
        help="Camera focal length x (default: 605.0)",
    )
    parser.add_argument(
        "--camera-fy",
        type=float,
        default=605.0,
        help="Camera focal length y (default: 605.0)",
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Gello PKL -> HDF5 Converter with Annotations")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Objects: {args.objects or 'None'}")
    print(f"Detect BBox: {args.detect_bbox}")
    print(f"Detect Mask: {args.detect_mask}")
    print(f"Estimate Pose: {args.estimate_pose}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    convert_gello_to_hdf(
        input_dir=args.input,
        output_file=args.output,
        objects=args.objects,
        mesh_dir=args.mesh_dir,
        detect_bbox=args.detect_bbox,
        detect_mask=args.detect_mask,
        estimate_pose=args.estimate_pose,
        grounding_dino_config=args.grounding_dino_config,
        grounding_dino_checkpoint=args.grounding_dino_checkpoint,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        foundationpose_score_ckpt=args.foundationpose_score_ckpt,
        foundationpose_refine_ckpt=args.foundationpose_refine_ckpt,
        camera_fx=args.camera_fx,
        camera_fy=args.camera_fy,
        device=args.device,
        overwrite=args.overwrite,
    )
    
    print("\n✓ Conversion completed successfully!")


if __name__ == "__main__":
    main()
