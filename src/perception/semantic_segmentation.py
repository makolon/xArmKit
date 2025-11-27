#!/usr/bin/env python3
"""
Semantic Segmentation with RealSense, GroundingDINO and SAM2

This script combines:
1. GroundingDINO for object detection (bounding boxes + labels)
2. SAM2 for instance segmentation based on detected boxes
3. RealSense for RGB-D capture
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

XARMKIT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(XARMKIT_ROOT / "src"))
sys.path.insert(0, str(XARMKIT_ROOT / "third_party" / "sam2"))

from real.realsense import RealSenseCamera
from utils.groundingdino_utils import GroundingDINODetector, detect_objects_from_instruction
from utils.sam_utils import SAM2Segmenter


class RealSenseSemanticSegmenter:
    """
    Combined detector and segmenter using RealSense + GroundingDINO + SAM2.
    
    Workflow:
    1. Capture RGB-D frames from RealSense
    2. Detect objects with GroundingDINO (bounding boxes + labels)
    3. Segment detected objects with SAM2 using bounding boxes as prompts
    """
    
    def __init__(
        self,
        camera: RealSenseCamera,
        detector: GroundingDINODetector,
        segmenter: SAM2Segmenter,
        enable_logging: bool = True,
    ):
        self.camera = camera
        self.detector = detector
        self.segmenter = segmenter
        self.enable_logging = enable_logging
        self._log("RealSenseSemanticSegmenter initialized")
    
    def _log(self, message: str):
        if self.enable_logging:
            print(f"[SemanticSegmenter] {message}")
    
    def segment_objects(
        self,
        object_names: List[str],
        visualize: bool = False,
        return_depth: bool = False,
        confidence_threshold: float = 0.3,
    ) -> Dict:
        """
        Detect and segment objects in the current frame.
        
        Args:
            object_names: List of object names to detect
            visualize: Whether to show visualization
            return_depth: Whether to include depth information
            confidence_threshold: Minimum confidence for detection
        
        Returns:
            Dictionary containing detection and segmentation results for each object
        """
        # Capture frames
        frames = self.camera.get_frames(align_depth_to_color=True)
        if frames is None or 'color' not in frames:
            self._log("Failed to capture frames")
            return {}
        
        color_image = frames['color']
        depth_image = frames.get('depth') if return_depth else None
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        # Step 1: Detect objects with GroundingDINO
        self._log(f"Detecting objects: {object_names}")
        boxes_dict = detect_objects_from_instruction(
            rgb_image, 
            object_names, 
            self.detector, 
            return_best_only=True,  # Get best detection per object
        )
        
        # Step 2: Segment detected objects with SAM2
        self.segmenter.set_image(rgb_image)
        
        results = {}
        for obj_name in object_names:
            bbox = boxes_dict.get(obj_name)
            
            if bbox is None:
                results[obj_name] = {
                    "bbox": None,
                    "mask": None,
                    "score": 0.0,
                    "center_2d": None,
                }
                continue
            
            # Convert to numpy if needed
            bbox = bbox.cpu().numpy() if isinstance(bbox, torch.Tensor) else bbox
            
            x1, y1, x2, y2 = bbox
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Segment with SAM2 using the bounding box
            self._log(f"Segmenting '{obj_name}' with SAM2...")
            masks, scores, logits = self.segmenter.segment_from_box(
                box=bbox,
                multimask_output=False,
            )
            
            # Get the best mask
            best_mask = masks[0] if len(masks) > 0 else None
            mask_score = scores[0] if len(scores) > 0 else 0.0
            
            result = {
                "bbox": bbox,
                "mask": best_mask,
                "score": float(mask_score),
                "center_2d": (center_x, center_y),
            }
            
            # Add depth information if requested
            if return_depth and depth_image is not None and best_mask is not None:
                # Get depth at mask center
                depth_value = depth_image[center_y, center_x]
                
                if depth_value > 0:
                    depth_m = depth_value * 0.001
                    result["depth"] = depth_m
                    
                    # Deproject to 3D
                    point_3d = self.camera.deproject_pixel_to_point(
                        (center_x, center_y), 
                        depth_value, 
                        use_color_intrinsics=True
                    )
                    
                    if point_3d is not None:
                        result["center_3d"] = point_3d
                        self._log(
                            f"  '{obj_name}': 3D position: "
                            f"x={point_3d[0]:.3f}m, y={point_3d[1]:.3f}m, z={point_3d[2]:.3f}m"
                        )
                    
                    # Calculate average depth over the mask
                    mask_depth = depth_image[best_mask > 0]
                    valid_depth = mask_depth[mask_depth > 0]
                    if len(valid_depth) > 0:
                        result["avg_depth"] = float(np.mean(valid_depth) * 0.001)
                else:
                    result["depth"] = None
                    result["center_3d"] = None
            
            results[obj_name] = result
            self._log(f"  '{obj_name}': mask score={mask_score:.3f}, bbox={bbox}")
        
        if visualize:
            vis_image = self._visualize_results(color_image, results)
            cv2.imshow("Semantic Segmentation", vis_image)
            cv2.waitKey(1)
        
        return results
    
    def _visualize_results(
        self, 
        image: np.ndarray, 
        results: Dict,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Visualize segmentation results with colored masks and labels.
        
        Args:
            image: Original RGB image
            results: Segmentation results
            alpha: Transparency of mask overlay
        
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        # Generate consistent colors for each object
        np.random.seed(42)
        colors = {}
        for obj_name in results.keys():
            colors[obj_name] = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Draw masks
        for obj_name, result in results.items():
            if result["mask"] is None:
                continue
            
            mask = result["mask"]
            color = colors[obj_name]
            
            # Create colored mask overlay
            mask_colored = np.zeros_like(vis_image)
            mask_colored[mask > 0] = color
            
            # Blend with original image
            vis_image = cv2.addWeighted(vis_image, 1.0, mask_colored, alpha, 0)
            
            # Draw bounding box
            if result["bbox"] is not None:
                bbox = result["bbox"]
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label text
                label = obj_name
                if "depth" in result and result["depth"] is not None:
                    label += f" ({result['depth']:.2f}m)"
                if "score" in result:
                    label += f" [{result['score']:.2f}]"
                
                # Draw label background and text
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    vis_image, 
                    (x1, y1 - label_h - baseline - 5), 
                    (x1 + label_w, y1), 
                    color, 
                    -1
                )
                cv2.putText(
                    vis_image, 
                    label, 
                    (x1, y1 - baseline - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1
                )
                
                # Draw center point
                center = result["center_2d"]
                if center is not None:
                    cv2.circle(vis_image, center, 5, (0, 0, 255), -1)
        
        return vis_image
    
    def save_masks(
        self, 
        results: Dict, 
        output_dir: Path,
        save_individual: bool = True,
        save_combined: bool = True,
    ):
        """
        Save segmentation masks to files.
        
        Args:
            results: Segmentation results
            output_dir: Directory to save masks
            save_individual: Save individual masks for each object
            save_combined: Save combined mask with different labels
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_individual:
            for obj_name, result in results.items():
                if result["mask"] is not None:
                    mask_path = output_dir / f"{obj_name}_mask.png"
                    mask_uint8 = (result["mask"] * 255).astype(np.uint8)
                    cv2.imwrite(str(mask_path), mask_uint8)
                    self._log(f"Saved mask: {mask_path}")
        
        if save_combined:
            # Create combined label map
            frames = self.camera.get_frames()
            if frames is not None:
                h, w = frames['color'].shape[:2]
                label_map = np.zeros((h, w), dtype=np.uint8)
                
                for idx, (obj_name, result) in enumerate(results.items(), start=1):
                    if result["mask"] is not None:
                        label_map[result["mask"] > 0] = idx
                
                combined_path = output_dir / "combined_labels.png"
                cv2.imwrite(str(combined_path), label_map)
                self._log(f"Saved combined labels: {combined_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Segmentation with GroundingDINO + SAM2"
    )
    parser.add_argument(
        "--objects", 
        type=str, 
        nargs="+", 
        default=["banana", "peach"], 
        help="Object names to detect and segment"
    )
    parser.add_argument(
        "--grounding-config", 
        type=str, 
        default="third_party/groundingdino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="GroundingDINO config path"
    )
    parser.add_argument(
        "--grounding-checkpoint", 
        type=str, 
        default="third_party/groundingdino/weights/groundingdino_swint_ogc.pth",
        help="GroundingDINO checkpoint path"
    )
    parser.add_argument(
        "--sam-config", 
        type=str, 
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 config path (relative to sam2/ directory)"
    )
    parser.add_argument(
        "--sam-checkpoint", 
        type=str, 
        default="third_party/sam2/checkpoints/sam2.1_hiera_large.pt",
        help="SAM2 checkpoint path"
    )
    parser.add_argument(
        "--width", 
        type=int, 
        default=640,
        help="Camera width"
    )
    parser.add_argument(
        "--height", 
        type=int, 
        default=480,
        help="Camera height"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="./segmentation_output.png",
        help="Path to save visualization"
    )
    parser.add_argument(
        "--save-masks", 
        type=str,
        help="Directory to save individual masks"
    )
    parser.add_argument(
        "--headless", 
        action="store_true",
        help="Run without GUI display"
    )
    parser.add_argument(
        "--confidence", 
        type=float, 
        default=0.3,
        help="Detection confidence threshold"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("RealSense + GroundingDINO + SAM2 Semantic Segmentation")
    print("=" * 60)
    
    # Initialize RealSense camera
    print("\n[1/4] Initializing RealSense camera...")
    camera = RealSenseCamera(
        width=args.width, 
        height=args.height, 
        fps=30, 
        enable_rgb=True, 
        enable_depth=True
    )
    camera.start()
    
    # Initialize GroundingDINO
    print("\n[2/4] Initializing GroundingDINO detector...")
    detector = GroundingDINODetector(
        config_path=str(XARMKIT_ROOT / args.grounding_config),
        checkpoint_path=str(XARMKIT_ROOT / args.grounding_checkpoint),
    )
    
    # Initialize SAM2
    print("\n[3/4] Initializing SAM2 segmenter...")
    # SAM2 expects config path like "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_config = args.sam_config
    sam_checkpoint = str(XARMKIT_ROOT / args.sam_checkpoint)
    segmenter = SAM2Segmenter(
        checkpoint_path=sam_checkpoint,
        model_cfg=sam_config,
    )
    
    # Initialize combined segmenter
    print("\n[4/4] Initializing semantic segmenter...")
    semantic_segmenter = RealSenseSemanticSegmenter(
        camera=camera,
        detector=detector,
        segmenter=segmenter,
    )
    
    print(f"\nTarget objects: {args.objects}\n")
    
    # Run segmentation
    visualize = not args.headless
    results = semantic_segmenter.segment_objects(
        args.objects,
        visualize=visualize,
        return_depth=True,
        confidence_threshold=args.confidence,
    )
    
    # Print results
    print("\nSegmentation Results:")
    print("-" * 60)
    for obj_name, result in results.items():
        print(f"\n{obj_name}:")
        if result["mask"] is not None:
            mask_area = np.sum(result["mask"] > 0)
            print(f"  Mask area: {mask_area} pixels")
            print(f"  Mask score: {result['score']:.3f}")
            print(f"  BBox: {result['bbox']}")
            if "center_3d" in result and result["center_3d"] is not None:
                pos = result["center_3d"]
                print(f"  3D position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m")
        else:
            print("  Not detected")
    
    # Save visualization
    if args.output:
        frames = camera.get_frames()
        vis_image = semantic_segmenter._visualize_results(frames['color'], results)
        cv2.imwrite(args.output, vis_image)
        print(f"\nSaved visualization to {args.output}")
    
    # Save individual masks if requested
    if args.save_masks:
        semantic_segmenter.save_masks(
            results,
            Path(args.save_masks),
            save_individual=True,
            save_combined=True,
        )
    
    # Wait for key press if not headless
    if not args.headless:
        print("\nPress any key to exit...")
        cv2.waitKey(0)
    
    # Cleanup
    cv2.destroyAllWindows()
    camera.stop()
    print("\nDone!")


if __name__ == "__main__":
    main()
