"""
FoundationPose wrapper for 6D pose estimation.

This module provides a convenient wrapper around FoundationPose for pose estimation
and tracking from RGBD sequences.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import trimesh
from PIL import Image

# Add FoundationPose to path
FOUNDATION_POSE_PATH = Path(__file__).parent.parent.parent / "third_party" / "FoundationPose"
sys.path.insert(0, str(FOUNDATION_POSE_PATH))

from estimater import FoundationPose
from datareader import *
from Utils import *


class FoundationPoseEstimator:
    """
    Wrapper for FoundationPose to estimate 6D poses from RGBD sequences.
    
    This class simplifies the use of FoundationPose for object pose estimation
    and tracking across video frames.
    """
    
    def __init__(
        self,
        mesh_path: str,
        model_cfg_path: Optional[str] = None,
        score_checkpoint: Optional[str] = None,
        refine_checkpoint: Optional[str] = None,
        device: str = "cuda",
        debug: int = 0,
    ):
        """
        Initialize FoundationPose estimator.
        
        Args:
            mesh_path: Path to object mesh file (.obj, .ply, etc.)
            model_cfg_path: Path to model config YAML (optional)
            score_checkpoint: Path to score predictor checkpoint (optional)
            refine_checkpoint: Path to pose refiner checkpoint (optional)
            device: Device to run inference on
            debug: Debug level (0=off, 1=info, 2=verbose)
        """
        self.device = device
        self.debug = debug
        
        print(f"[FoundationPose] Loading mesh from {mesh_path}")
        
        # Load mesh
        mesh = trimesh.load(mesh_path)
        
        # Extract model points and normals
        model_pts = np.asarray(mesh.vertices, dtype=np.float32)
        
        # Compute normals if not available
        if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
            mesh.compute_vertex_normals()
        model_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        
        # Load score predictor
        scorer = None
        if score_checkpoint is not None:
            from learning.training.predict_score import ScorePredictor
            scorer = ScorePredictor()
            scorer.load_state_dict(torch.load(score_checkpoint))
            scorer.eval()
            scorer = scorer.to(device)
        
        # Load pose refiner
        refiner = None
        if refine_checkpoint is not None:
            from learning.training.predict_pose_refine import PoseRefinePredictor
            refiner = PoseRefinePredictor()
            refiner.load_state_dict(torch.load(refine_checkpoint))
            refiner.eval()
            refiner = refiner.to(device)
        
        # Initialize FoundationPose
        print(f"[FoundationPose] Initializing pose estimator")
        self.estimator = FoundationPose(
            model_pts=model_pts,
            model_normals=model_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug=debug,
        )
        
        self.mesh = mesh
        self.is_tracking = False
        
        print(f"[FoundationPose] Initialization complete")
    
    def register(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        iteration: int = 5,
    ) -> np.ndarray:
        """
        Register object pose from single RGBD frame.
        
        Args:
            rgb: RGB image (H, W, 3) uint8
            depth: Depth image (H, W) in meters
            mask: Object mask (H, W) binary
            K: Camera intrinsic matrix (3, 3)
            iteration: Number of refinement iterations
        
        Returns:
            Pose matrix (4, 4) in camera frame
        """
        # Ensure correct dtypes
        rgb = rgb.astype(np.uint8)
        depth = depth.astype(np.float32)
        mask = mask.astype(np.uint8)
        K = K.astype(np.float32)
        
        # Register pose
        pose = self.estimator.register(
            K=K,
            rgb=rgb,
            depth=depth,
            ob_mask=mask,
            iteration=iteration,
        )
        
        self.is_tracking = True
        return pose
    
    def track(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        iteration: int = 2,
    ) -> np.ndarray:
        """
        Track object pose using previous pose as prior.
        
        Args:
            rgb: RGB image (H, W, 3) uint8
            depth: Depth image (H, W) in meters
            K: Camera intrinsic matrix (3, 3)
            iteration: Number of refinement iterations
        
        Returns:
            Pose matrix (4, 4) in camera frame
        """
        if not self.is_tracking:
            raise RuntimeError("Must call register() before track()")
        
        # Ensure correct dtypes
        rgb = rgb.astype(np.uint8)
        depth = depth.astype(np.float32)
        K = K.astype(np.float32)
        
        # Track pose
        pose = self.estimator.track_one(
            rgb=rgb,
            depth=depth,
            K=K,
            iteration=iteration,
        )
        
        return pose
    
    def reset(self):
        """Reset tracking state."""
        self.is_tracking = False
        self.estimator.pose_last = None
    
    @staticmethod
    def pose_to_position_quaternion(pose: np.ndarray) -> np.ndarray:
        """
        Convert pose matrix to position + quaternion representation.
        
        Args:
            pose: Pose matrix (4, 4)
        
        Returns:
            Array of [x, y, z, qw, qx, qy, qz]
        """
        from scipy.spatial.transform import Rotation
        
        position = pose[:3, 3]
        rotation = Rotation.from_matrix(pose[:3, :3])
        quaternion = rotation.as_quat()  # [x, y, z, w]
        
        # Convert to [w, x, y, z] format
        quat_wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
        
        return np.concatenate([position, quat_wxyz])


def estimate_poses_from_sequence(
    rgb_sequence: List[np.ndarray],
    depth_sequence: List[np.ndarray],
    masks_sequence: List[np.ndarray],
    K: np.ndarray,
    mesh_path: str,
    use_tracking: bool = True,
    register_interval: int = 10,
    score_checkpoint: Optional[str] = None,
    refine_checkpoint: Optional[str] = None,
    device: str = "cuda",
) -> List[np.ndarray]:
    """
    Estimate object poses from RGBD sequence.
    
    Args:
        rgb_sequence: List of RGB images (H, W, 3)
        depth_sequence: List of depth images (H, W)
        masks_sequence: List of object masks (H, W)
        K: Camera intrinsic matrix (3, 3)
        mesh_path: Path to object mesh file
        use_tracking: Whether to use tracking between frames
        register_interval: Re-register every N frames (if tracking)
        score_checkpoint: Path to score predictor checkpoint
        refine_checkpoint: Path to pose refiner checkpoint
        device: Device to run on
    
    Returns:
        List of pose matrices (4, 4) for each frame
    """
    estimator = FoundationPoseEstimator(
        mesh_path=mesh_path,
        score_checkpoint=score_checkpoint,
        refine_checkpoint=refine_checkpoint,
        device=device,
    )
    
    poses = []
    
    for frame_idx, (rgb, depth, mask) in enumerate(
        zip(rgb_sequence, depth_sequence, masks_sequence)
    ):
        # Register on first frame or at intervals
        if frame_idx == 0 or not use_tracking or frame_idx % register_interval == 0:
            pose = estimator.register(rgb, depth, mask, K)
            print(f"[FoundationPose] Frame {frame_idx}: Registered")
        else:
            pose = estimator.track(rgb, depth, K)
            print(f"[FoundationPose] Frame {frame_idx}: Tracked")
        
        poses.append(pose)
    
    return poses


def estimate_multi_object_poses(
    rgb_sequence: List[np.ndarray],
    depth_sequence: List[np.ndarray],
    masks_dict_sequence: List[Dict[str, np.ndarray]],
    K: np.ndarray,
    mesh_paths: Dict[str, str],
    use_tracking: bool = True,
    register_interval: int = 10,
    score_checkpoint: Optional[str] = None,
    refine_checkpoint: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, List[np.ndarray]]:
    """
    Estimate poses for multiple objects from RGBD sequence.
    
    Args:
        rgb_sequence: List of RGB images (H, W, 3)
        depth_sequence: List of depth images (H, W)
        masks_dict_sequence: List of dicts mapping object names to masks
        K: Camera intrinsic matrix (3, 3)
        mesh_paths: Dict mapping object names to mesh file paths
        use_tracking: Whether to use tracking between frames
        register_interval: Re-register every N frames (if tracking)
        score_checkpoint: Path to score predictor checkpoint
        refine_checkpoint: Path to pose refiner checkpoint
        device: Device to run on
    
    Returns:
        Dict mapping object names to lists of pose matrices
    """
    # Initialize estimators for each object
    estimators = {}
    for obj_name, mesh_path in mesh_paths.items():
        print(f"\n[FoundationPose] Initializing estimator for '{obj_name}'")
        estimators[obj_name] = FoundationPoseEstimator(
            mesh_path=mesh_path,
            score_checkpoint=score_checkpoint,
            refine_checkpoint=refine_checkpoint,
            device=device,
        )
    
    # Track poses for each object
    poses_dict = {obj_name: [] for obj_name in mesh_paths.keys()}
    
    for frame_idx, (rgb, depth, masks_dict) in enumerate(
        zip(rgb_sequence, depth_sequence, masks_dict_sequence)
    ):
        print(f"\n[FoundationPose] Processing frame {frame_idx}/{len(rgb_sequence)}")
        
        for obj_name, estimator in estimators.items():
            if obj_name not in masks_dict:
                # No mask for this object in this frame - use identity or skip
                poses_dict[obj_name].append(np.eye(4))
                print(f"  {obj_name}: No mask, using identity pose")
                continue
            
            mask = masks_dict[obj_name]
            
            # Register on first frame or at intervals
            if frame_idx == 0 or not use_tracking or frame_idx % register_interval == 0:
                pose = estimator.register(rgb, depth, mask, K)
                print(f"  {obj_name}: Registered")
            else:
                pose = estimator.track(rgb, depth, K)
                print(f"  {obj_name}: Tracked")
            
            poses_dict[obj_name].append(pose)
    
    return poses_dict
