#!/usr/bin/env python3
"""
Bounding Box Detection with RealSense and GroundingDINO
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

XARMKIT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(XARMKIT_ROOT / "src"))

from real.realsense import RealSenseCamera
from utils.groundingdino_utils import GroundingDINODetector, detect_objects_from_instruction


class RealSenseBBoxDetector:
    def __init__(self, camera: RealSenseCamera, detector: GroundingDINODetector, enable_logging: bool = True):
        self.camera = camera
        self.detector = detector
        self.enable_logging = enable_logging
        self._log("RealSenseBBoxDetector initialized")
    
    def _log(self, message: str):
        if self.enable_logging:
            print(f"[BBoxDetector] {message}")
    
    def detect_objects(self, object_names: List[str], visualize: bool = False, return_depth: bool = False) -> Dict:
        frames = self.camera.get_frames(align_depth_to_color=True)
        if frames is None or 'color' not in frames:
            self._log("Failed to capture frames")
            return {}
        
        color_image = frames['color']
        depth_image = frames.get('depth') if return_depth else None
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        self._log(f"Detecting objects: {object_names}")
        boxes_dict = detect_objects_from_instruction(rgb_image, object_names, self.detector, return_best_only=True)
        
        results = {}
        for obj_name in object_names:
            bbox = boxes_dict.get(obj_name)
            
            if bbox is not None:
                bbox_np = bbox.cpu().numpy() if isinstance(bbox, torch.Tensor) else bbox
                x1, y1, x2, y2 = bbox_np
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                result = {"bbox": bbox_np, "score": 1.0, "center_2d": (center_x, center_y)}
                
                if return_depth and depth_image is not None:
                    depth_value = depth_image[center_y, center_x]
                    if depth_value > 0:
                        depth_m = depth_value * 0.001
                        result["depth"] = depth_m
                        point_3d = self.camera.deproject_pixel_to_point((center_x, center_y), depth_value, use_color_intrinsics=True)
                        if point_3d is not None:
                            result["center_3d"] = point_3d
                            self._log(f"  3D position: x={point_3d[0]:.3f}m, y={point_3d[1]:.3f}m, z={point_3d[2]:.3f}m")
                    else:
                        result["depth"] = None
                        result["center_3d"] = None
                
                results[obj_name] = result
            else:
                results[obj_name] = {"bbox": None, "score": 0.0, "center_2d": None}
        
        if visualize:
            vis_image = self._visualize_results(color_image, results)
            cv2.imshow("Object Detection", vis_image)
            cv2.waitKey(1)
        
        return results
    
    def _visualize_results(self, image: np.ndarray, results: Dict) -> np.ndarray:
        vis_image = image.copy()
        for obj_name, result in results.items():
            if result["bbox"] is not None:
                bbox = result["bbox"]
                x1, y1, x2, y2 = bbox.astype(int)
                color = (0, 255, 0)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                label = obj_name
                if "depth" in result and result["depth"] is not None:
                    label += f" ({result['depth']:.2f}m)"
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis_image, (x1, y1 - label_h - baseline - 5), (x1 + label_w, y1), color, -1)
                cv2.putText(vis_image, label, (x1, y1 - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                center = result["center_2d"]
                if center is not None:
                    cv2.circle(vis_image, center, 5, (0, 0, 255), -1)
        return vis_image
    
    def continuous_detection(self, object_names: List[str], update_rate_hz: float = 10.0, show_depth: bool = True):
        self._log(f"Starting continuous detection at {update_rate_hz} Hz. Press 'q' to quit")
        interval = 1.0 / update_rate_hz
        
        while True:
            start_time = time.time()
            self.detect_objects(object_names, visualize=True, return_depth=show_depth)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                self._log("Continuous detection stopped")
                break
            elapsed = time.time() - start_time
            if elapsed < interval:
                time.sleep(interval - elapsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", type=str, nargs="+", default=["banana", "peach"], help="Object names")
    parser.add_argument("--config", type=str, default="third_party/groundingdino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--checkpoint", type=str, default="third_party/groundingdino/weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--output", type=str, default="./detection_output.png", help="Path to save the output image")
    parser.add_argument("--headless", action="store_true", help="Run without GUI display")
    args = parser.parse_args()
    
    print("="*60)
    print("RealSense + GroundingDINO Object Detection")
    print("="*60)
    
    print("\n[1/3] Initializing RealSense camera...")
    camera = RealSenseCamera(width=args.width, height=args.height, fps=30, enable_rgb=True, enable_depth=True)
    camera.start()
    
    print("\n[2/3] Initializing GroundingDINO detector...")
    detector = GroundingDINODetector(config_path=str(XARMKIT_ROOT/args.config), checkpoint_path=str(XARMKIT_ROOT/args.checkpoint))
    
    print("\n[3/3] Initializing combined detector...")
    bbox_detector = RealSenseBBoxDetector(camera, detector)
    
    print(f"\nTarget objects: {args.objects}\n")
    
    if args.continuous:
        bbox_detector.continuous_detection(args.objects)
    else:
        visualize = not args.headless
        results = bbox_detector.detect_objects(args.objects, visualize=visualize, return_depth=True)
        print("\nDetection Results:")
        for obj_name, result in results.items():
            print(f"{obj_name}: {result}")
        if args.output:
            frames = camera.get_frames()
            vis_image = bbox_detector._visualize_results(frames['color'], results)
            cv2.imwrite(args.output, vis_image)
            print(f"Saved to {args.output}")
        elif not args.headless:
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    camera.stop()
    print("\nDone!")


if __name__ == "__main__":
    main()
