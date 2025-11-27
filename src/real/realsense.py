#!/usr/bin/env python3
"""
RealSense Camera Interface

This module provides a high-level interface for Intel RealSense cameras.
It supports RGB and depth image capture with configurable resolution and framerate.
"""

import numpy as np
import pyrealsense2 as rs
from typing import Optional, Tuple, Dict, Any
import time


class RealSenseCamera:
    """
    High-level controller for Intel RealSense cameras (D400 series).
    
    This class wraps pyrealsense2 to provide convenient methods for:
    - Camera initialization and configuration
    - RGB and depth image capture
    - Intrinsic camera parameters retrieval
    - Aligned depth to color frames
    """
    
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_rgb: bool = True,
        enable_depth: bool = True,
        enable_logging: bool = True,
        device_serial: Optional[str] = None,
    ):
        """
        Initialize RealSense camera.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            fps: Frames per second
            enable_rgb: Enable RGB stream
            enable_depth: Enable depth stream
            enable_logging: Enable verbose logging
            device_serial: Specific device serial number (None for first available)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth
        self.enable_logging = enable_logging
        self.device_serial = device_serial
        
        # Initialize RealSense pipeline and config
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Camera intrinsics (will be populated after start)
        self.color_intrinsics: Optional[rs.intrinsics] = None
        self.depth_intrinsics: Optional[rs.intrinsics] = None
        self.depth_scale: Optional[float] = None
        
        # Alignment object for aligning depth to color
        self.align = None
        
        # Pipeline profile
        self.profile = None
        
        self._log("RealSense camera initialized")
    
    def _log(self, message: str):
        """Internal logging method."""
        if self.enable_logging:
            print(f"[RealSense] {message}")
    
    def list_devices(self) -> list:
        """
        List all connected RealSense devices.
        
        Returns:
            List of device serial numbers
        """
        ctx = rs.context()
        devices = ctx.query_devices()
        device_list = []
        
        for dev in devices:
            serial = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)
            device_list.append({"serial": serial, "name": name})
            self._log(f"Found device: {name} (Serial: {serial})")
        
        return device_list
    
    def start(self) -> bool:
        """
        Start the camera streaming.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Configure device if serial number is specified
            if self.device_serial is not None:
                self.config.enable_device(self.device_serial)
                self._log(f"Using device with serial: {self.device_serial}")
            
            # Configure streams
            if self.enable_rgb:
                self.config.enable_stream(
                    rs.stream.color, 
                    self.width, 
                    self.height, 
                    rs.format.bgr8, 
                    self.fps
                )
                self._log(f"RGB stream enabled: {self.width}x{self.height} @ {self.fps}fps")
            
            if self.enable_depth:
                self.config.enable_stream(
                    rs.stream.depth, 
                    self.width, 
                    self.height, 
                    rs.format.z16, 
                    self.fps
                )
                self._log(f"Depth stream enabled: {self.width}x{self.height} @ {self.fps}fps")
            
            # Start streaming
            self.profile = self.pipeline.start(self.config)
            
            # Get camera intrinsics and depth scale
            if self.enable_rgb:
                color_stream = self.profile.get_stream(rs.stream.color)
                self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
                self._log(f"Color intrinsics: fx={self.color_intrinsics.fx:.2f}, "
                         f"fy={self.color_intrinsics.fy:.2f}, "
                         f"cx={self.color_intrinsics.ppx:.2f}, "
                         f"cy={self.color_intrinsics.ppy:.2f}")
            
            if self.enable_depth:
                depth_stream = self.profile.get_stream(rs.stream.depth)
                self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
                
                # Get depth scale
                depth_sensor = self.profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
                self._log(f"Depth scale: {self.depth_scale}")
            
            # Create alignment object (depth to color)
            if self.enable_rgb and self.enable_depth:
                self.align = rs.align(rs.stream.color)
            
            # Wait for camera to stabilize
            self._log("Warming up camera...")
            for _ in range(30):
                self.pipeline.wait_for_frames()
            
            self._log("Camera started successfully!")
            return True
            
        except Exception as e:
            self._log(f"Error starting camera: {e}")
            return False
    
    def get_frames(
        self, 
        align_depth_to_color: bool = True
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Capture a single frame from the camera.
        
        Args:
            align_depth_to_color: If True, align depth frame to color frame
        
        Returns:
            Dictionary containing:
            - 'color': RGB image as numpy array (H, W, 3) in BGR format
            - 'depth': Depth image as numpy array (H, W) in millimeters
            Returns None if capture fails
        """
        try:
            # Wait for a coherent pair of frames
            frames = self.pipeline.wait_for_frames()
            
            # Align depth to color if requested
            if align_depth_to_color and self.align is not None:
                frames = self.align.process(frames)
            
            result = {}
            
            # Get color frame
            if self.enable_rgb:
                color_frame = frames.get_color_frame()
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    result['color'] = color_image
            
            # Get depth frame
            if self.enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    result['depth'] = depth_image
            
            return result if result else None
            
        except Exception as e:
            self._log(f"Error capturing frames: {e}")
            return None
    
    def get_color_image(self) -> Optional[np.ndarray]:
        """
        Get only the color image.
        
        Returns:
            RGB image as numpy array (H, W, 3) in BGR format, or None if failed
        """
        frames = self.get_frames()
        return frames.get('color') if frames else None
    
    def get_depth_image(self) -> Optional[np.ndarray]:
        """
        Get only the depth image.
        
        Returns:
            Depth image as numpy array (H, W) in millimeters, or None if failed
        """
        frames = self.get_frames()
        return frames.get('depth') if frames else None
    
    def get_rgbd(self, align: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get aligned RGB and depth images.
        
        Args:
            align: Whether to align depth to color
        
        Returns:
            Tuple of (color_image, depth_image) or None if failed
        """
        frames = self.get_frames(align_depth_to_color=align)
        if frames and 'color' in frames and 'depth' in frames:
            return frames['color'], frames['depth']
        return None
    
    def deproject_pixel_to_point(
        self, 
        pixel: Tuple[int, int], 
        depth_value: float,
        use_color_intrinsics: bool = True
    ) -> Optional[Tuple[float, float, float]]:
        """
        Convert a 2D pixel coordinate to 3D point in camera frame.
        
        Args:
            pixel: (x, y) pixel coordinates
            depth_value: Depth value at that pixel (in millimeters)
            use_color_intrinsics: If True, use color intrinsics; else depth intrinsics
        
        Returns:
            (x, y, z) 3D point in meters, or None if intrinsics not available
        """
        intrinsics = self.color_intrinsics if use_color_intrinsics else self.depth_intrinsics
        
        if intrinsics is None:
            self._log("Camera intrinsics not available")
            return None
        
        # Convert depth to meters
        depth_m = depth_value * 0.001  # mm to m
        
        # Deproject
        point = rs.rs2_deproject_pixel_to_point(intrinsics, list(pixel), depth_m)
        return tuple(point)
    
    def get_intrinsics_matrix(
        self, 
        use_color: bool = True
    ) -> Optional[np.ndarray]:
        """
        Get camera intrinsic matrix in standard format.
        
        Args:
            use_color: If True, return color intrinsics; else depth intrinsics
        
        Returns:
            3x3 intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        """
        intrinsics = self.color_intrinsics if use_color else self.depth_intrinsics
        
        if intrinsics is None:
            return None
        
        K = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        return K
    
    def get_distortion_coefficients(
        self, 
        use_color: bool = True
    ) -> Optional[np.ndarray]:
        """
        Get camera distortion coefficients.
        
        Args:
            use_color: If True, return color distortion; else depth distortion
        
        Returns:
            Distortion coefficients array
        """
        intrinsics = self.color_intrinsics if use_color else self.depth_intrinsics
        
        if intrinsics is None:
            return None
        
        return np.array(intrinsics.coeffs)
    
    def stop(self):
        """Stop the camera streaming."""
        try:
            self.pipeline.stop()
            self._log("Camera stopped")
        except Exception as e:
            self._log(f"Error stopping camera: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def __del__(self):
        """Destructor."""
        try:
            self.stop()
        except:
            pass


if __name__ == "__main__":
    # Example usage
    print("RealSense Camera Test")
    print("=" * 50)
    
    # List available devices
    camera = RealSenseCamera()
    devices = camera.list_devices()
    print(f"\nFound {len(devices)} device(s)")
    
    # Use context manager for automatic cleanup
    with RealSenseCamera(width=640, height=480, fps=30) as cam:
        print("\nCapturing frames...")
        
        # Capture a few frames
        for i in range(5):
            frames = cam.get_frames()
            if frames:
                if 'color' in frames:
                    print(f"Frame {i}: Color image shape: {frames['color'].shape}")
                if 'depth' in frames:
                    print(f"Frame {i}: Depth image shape: {frames['depth'].shape}")
            time.sleep(0.1)
        
        # Get intrinsics
        K = cam.get_intrinsics_matrix()
        if K is not None:
            print("\nCamera Intrinsic Matrix:")
            print(K)
        
        # Get a single color image
        print("\nCapturing single color image...")
        color_img = cam.get_color_image()
        if color_img is not None:
            print(f"Color image captured: {color_img.shape}")
    
    print("\nDone!")
