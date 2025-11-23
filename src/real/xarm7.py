#!/usr/bin/env python3
"""
xArm7 Robot Controller Class

This module provides a high-level interface for controlling the xArm7 robot.
It includes initialization, calibration, and control methods.
"""

import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from xarm.wrapper import XArmAPI


class XArm7:
    """
    High-level controller for xArm7 robot with initialization and calibration support.
    
    This class wraps the xArm-Python-SDK to provide convenient methods for:
    - Robot initialization and homing
    - Camera calibration
    - Position and joint control
    - Error handling and state management
    """
    
    # Default home position (joint angles in degrees)
    DEFAULT_HOME_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Default initial Cartesian position [x, y, z, roll, pitch, yaw]
    # This is approximately the home position in Cartesian coordinates
    DEFAULT_CARTESIAN_HOME = [201.5, 0.0, 140.5, 180.0, 0.0, 0.0]
    
    # Motion parameters
    DEFAULT_SPEED = 100  # mm/s for Cartesian, °/s for joint
    DEFAULT_ACC = 2000   # mm/s² for Cartesian, °/s² for joint
    
    def __init__(
        self,
        ip: str,
        is_radian: bool = False,
        home_position: Optional[List[float]] = None,
        enable_logging: bool = True
    ):
        """
        Initialize the xArm7 controller.
        
        Args:
            ip: IP address of the xArm controller (e.g., '192.168.1.194')
            is_radian: If True, use radians for angles; otherwise use degrees
            home_position: Custom home position (joint angles). If None, uses default
            enable_logging: Enable verbose logging
        """
        self.ip = ip
        self.is_radian = is_radian
        self.home_position = home_position or self.DEFAULT_HOME_POSITION
        self.enable_logging = enable_logging
        
        # Initialize the xArm API
        self.arm = XArmAPI(ip, is_radian=is_radian, do_not_open=False)
        
        # Camera calibration data
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.hand_eye_matrix: Optional[np.ndarray] = None
        
        self._log("XArm7 initialized with IP: {}".format(ip))
    
    def _log(self, message: str):
        """Internal logging method."""
        if self.enable_logging:
            print("[XArm7] {}".format(message))
    
    def _check_code(self, code: int, operation: str = "") -> bool:
        """
        Check the return code from xArm API calls.
        
        Args:
            code: Return code from API
            operation: Description of the operation for error messages
            
        Returns:
            True if successful (code == 0), False otherwise
        """
        if code == 0:
            return True
        else:
            error_msg = "Error in {}: code={}".format(operation, code)
            self._log(error_msg)
            return False
    
    def initialize(
        self,
        enable_motion: bool = True,
        set_mode: int = 0,
        set_state: int = 0,
        go_home: bool = True,
        speed: Optional[float] = None,
        acc: Optional[float] = None
    ) -> bool:
        """
        Initialize the robot to ready state and optionally move to home position.
        
        This method performs the following steps:
        1. Clear any errors
        2. Enable motion
        3. Set control mode (default: position control)
        4. Set state (default: ready)
        5. Optionally move to home position
        
        Args:
            enable_motion: Enable robot motion
            set_mode: Control mode (0=position, 1=servo, 2=joint teaching, etc.)
            set_state: Robot state (0=ready)
            go_home: Move to home position after initialization
            speed: Motion speed (uses default if None)
            acc: Motion acceleration (uses default if None)
            
        Returns:
            True if initialization successful, False otherwise
        """
        self._log("Starting initialization sequence...")
        
        # Clear errors and warnings
        self._log("Clearing errors and warnings...")
        code = self.arm.clean_error()
        if not self._check_code(code, "clean_error"):
            return False
        
        code = self.arm.clean_warn()
        if not self._check_code(code, "clean_warn"):
            return False
        
        # Enable motion
        if enable_motion:
            self._log("Enabling motion...")
            code = self.arm.motion_enable(enable=True)
            if not self._check_code(code, "motion_enable"):
                return False
        
        # Set mode
        self._log("Setting mode to {}...".format(set_mode))
        code = self.arm.set_mode(mode=set_mode)
        if not self._check_code(code, "set_mode"):
            return False
        
        # Set state
        self._log("Setting state to {}...".format(set_state))
        code = self.arm.set_state(state=set_state)
        if not self._check_code(code, "set_state"):
            return False
        
        # Wait for robot to be ready
        time.sleep(1)
        
        # Move to home position
        if go_home:
            self._log("Moving to home position...")
            success = self.move_to_home(speed=speed, acc=acc)
            if not success:
                return False
        
        self._log("Initialization complete!")
        return True
    
    def move_to_home(
        self,
        speed: Optional[float] = None,
        acc: Optional[float] = None,
        wait: bool = True
    ) -> bool:
        """
        Move the robot to the home position.
        
        Args:
            speed: Joint speed (°/s or rad/s depending on is_radian)
            acc: Joint acceleration (°/s² or rad/s² depending on is_radian)
            wait: Wait for motion to complete
            
        Returns:
            True if successful, False otherwise
        """
        speed = speed or self.DEFAULT_SPEED
        acc = acc or self.DEFAULT_ACC
        
        self._log("Moving to home position: {}".format(self.home_position))
        
        # Use set_servo_angle to move to home position
        code = self.arm.set_servo_angle(
            angle=self.home_position,
            speed=speed,
            mvacc=acc,
            wait=wait
        )
        
        return self._check_code(code, "move_to_home")
    
    def move_to_joint_position(
        self,
        angles: List[float],
        speed: Optional[float] = None,
        acc: Optional[float] = None,
        wait: bool = False
    ) -> bool:
        """
        Move to specified joint angles.
        
        Args:
            angles: List of 7 joint angles
            speed: Joint speed (°/s or rad/s)
            acc: Joint acceleration (°/s² or rad/s²)
            wait: Wait for motion to complete
            
        Returns:
            True if successful, False otherwise
        """
        if len(angles) != 7:
            self._log("Error: Expected 7 joint angles, got {}".format(len(angles)))
            return False
        
        speed = speed or self.DEFAULT_SPEED
        acc = acc or self.DEFAULT_ACC
        
        code = self.arm.set_servo_angle(
            angle=angles,
            speed=speed,
            mvacc=acc,
            wait=wait
        )
        
        return self._check_code(code, "move_to_joint_position")
    
    def move_to_cartesian_position(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
        speed: Optional[float] = None,
        acc: Optional[float] = None,
        wait: bool = False
    ) -> bool:
        """
        Move to specified Cartesian position.
        
        Args:
            x: X coordinate (mm)
            y: Y coordinate (mm)
            z: Z coordinate (mm)
            roll: Roll angle (° or rad)
            pitch: Pitch angle (° or rad)
            yaw: Yaw angle (° or rad)
            speed: Cartesian speed (mm/s)
            acc: Cartesian acceleration (mm/s²)
            wait: Wait for motion to complete
            
        Returns:
            True if successful, False otherwise
        """
        speed = speed or self.DEFAULT_SPEED
        acc = acc or self.DEFAULT_ACC
        
        code = self.arm.set_position(
            x=x, y=y, z=z,
            roll=roll, pitch=pitch, yaw=yaw,
            speed=speed,
            mvacc=acc,
            wait=wait
        )
        
        return self._check_code(code, "move_to_cartesian_position")
    
    def get_position(self) -> Tuple[bool, Optional[List[float]]]:
        """
        Get current Cartesian position.
        
        Returns:
            Tuple of (success, position) where position is [x, y, z, roll, pitch, yaw]
        """
        code, pos = self.arm.get_position(is_radian=self.is_radian)
        
        if self._check_code(code, "get_position"):
            return True, pos
        return False, None
    
    def get_joint_angles(self) -> Tuple[bool, Optional[List[float]]]:
        """
        Get current joint angles.
        
        Returns:
            Tuple of (success, angles) where angles is a list of 7 joint angles
        """
        code, angles = self.arm.get_servo_angle(is_radian=self.is_radian)
        
        if self._check_code(code, "get_joint_angles"):
            return True, angles
        return False, None
    
    def calibrate_camera_intrinsics(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ):
        """
        Set camera intrinsic calibration parameters.
        
        Args:
            camera_matrix: 3x3 camera matrix (fx, fy, cx, cy)
            dist_coeffs: Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self._log("Camera intrinsics calibrated")
    
    def calibrate_hand_eye(
        self,
        robot_poses: List[List[float]],
        camera_poses: List[np.ndarray],
        method: str = 'tsai'
    ) -> bool:
        """
        Perform hand-eye calibration using robot and camera poses.
        
        This method computes the transformation matrix between the robot's
        end-effector and the camera coordinate system.
        
        Args:
            robot_poses: List of robot TCP poses [x, y, z, roll, pitch, yaw]
            camera_poses: List of 4x4 camera transformation matrices
            method: Calibration method ('tsai', 'park', 'horaud', 'andreff', 'daniilidis')
            
        Returns:
            True if calibration successful, False otherwise
        """
        try:
            import cv2
            
            if len(robot_poses) != len(camera_poses):
                self._log("Error: Number of robot poses must match camera poses")
                return False
            
            if len(robot_poses) < 3:
                self._log("Error: At least 3 pose pairs required for calibration")
                return False
            
            # Convert robot poses to transformation matrices
            R_gripper2base = []
            t_gripper2base = []
            
            for pose in robot_poses:
                # Get rotation and translation from pose
                x, y, z = pose[0:3]
                roll, pitch, yaw = pose[3:6]
                
                # Convert to rotation matrix (simplified - use proper conversion)
                # This is a placeholder - implement proper Euler to rotation matrix
                R = np.eye(3)  # Replace with actual conversion
                t = np.array([[x], [y], [z]])
                
                R_gripper2base.append(R)
                t_gripper2base.append(t)
            
            # Extract rotation and translation from camera poses
            R_target2cam = []
            t_target2cam = []
            
            for cam_pose in camera_poses:
                R_target2cam.append(cam_pose[0:3, 0:3])
                t_target2cam.append(cam_pose[0:3, 3:4])
            
            # Perform calibration
            method_map = {
                'tsai': cv2.CALIB_HAND_EYE_TSAI,
                'park': cv2.CALIB_HAND_EYE_PARK,
                'horaud': cv2.CALIB_HAND_EYE_HORAUD,
                'andreff': cv2.CALIB_HAND_EYE_ANDREFF,
                'daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS
            }
            
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base,
                t_gripper2base,
                R_target2cam,
                t_target2cam,
                method=method_map.get(method, cv2.CALIB_HAND_EYE_TSAI)
            )
            
            # Store as 4x4 transformation matrix
            self.hand_eye_matrix = np.eye(4)
            self.hand_eye_matrix[0:3, 0:3] = R_cam2gripper
            self.hand_eye_matrix[0:3, 3:4] = t_cam2gripper
            
            self._log("Hand-eye calibration complete using {} method".format(method))
            return True
            
        except ImportError:
            self._log("Error: OpenCV required for hand-eye calibration")
            return False
        except Exception as e:
            self._log("Error during hand-eye calibration: {}".format(str(e)))
            return False
    
    def get_hand_eye_matrix(self) -> Optional[np.ndarray]:
        """
        Get the hand-eye calibration matrix.
        
        Returns:
            4x4 transformation matrix from camera to gripper, or None if not calibrated
        """
        return self.hand_eye_matrix
    
    def emergency_stop(self):
        """Emergency stop the robot."""
        self._log("EMERGENCY STOP!")
        self.arm.emergency_stop()
    
    def set_gripper(
        self,
        position: float,
        speed: Optional[float] = None,
        wait: bool = True
    ) -> bool:
        """
        Control the gripper.
        
        Args:
            position: Gripper position (0-850 for xArm gripper)
            speed: Gripper speed
            wait: Wait for gripper motion to complete
            
        Returns:
            True if successful, False otherwise
        """
        speed = speed or 5000
        
        code = self.arm.set_gripper_position(
            pos=position,
            speed=speed,
            wait=wait
        )
        
        return self._check_code(code, "set_gripper")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get comprehensive robot state information.
        
        Returns:
            Dictionary containing robot state information
        """
        state = {
            'connected': self.arm.connected,
            'mode': self.arm.mode,
            'state': self.arm.state,
            'error_code': self.arm.error_code,
            'warn_code': self.arm.warn_code,
        }
        
        # Get position
        success, pos = self.get_position()
        if success:
            state['position'] = pos
        
        # Get joint angles
        success, angles = self.get_joint_angles()
        if success:
            state['joint_angles'] = angles
        
        return state
    
    def reset(self, speed: Optional[float] = None, acc: Optional[float] = None) -> bool:
        """
        Reset the robot (clear errors and go to home).
        
        Args:
            speed: Motion speed
            acc: Motion acceleration
            
        Returns:
            True if successful, False otherwise
        """
        self._log("Resetting robot...")
        
        speed = speed or self.DEFAULT_SPEED
        acc = acc or self.DEFAULT_ACC
        
        code = self.arm.reset(speed=speed, mvacc=acc, wait=True)
        
        return self._check_code(code, "reset")
    
    def disconnect(self):
        """Disconnect from the robot."""
        self._log("Disconnecting from robot...")
        self.arm.disconnect()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures proper disconnection."""
        self.disconnect()
    
    def __del__(self):
        """Destructor - ensures proper disconnection."""
        try:
            self.disconnect()
        except:
            pass


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Get IP from command line or use default
    robot_ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.194"
    
    print("Connecting to xArm7 at {}...".format(robot_ip))
    
    # Use context manager for automatic cleanup
    with XArm7(robot_ip, is_radian=False) as robot:
        # Initialize robot
        if robot.initialize(go_home=True):
            print("Robot initialized successfully!")
            
            # Get current state
            state = robot.get_state()
            print("Robot state:", state)
            
            # Example: Move to a position
            print("Moving to test position...")
            robot.move_to_cartesian_position(
                x=300, y=0, z=200,
                roll=180, pitch=0, yaw=0,
                wait=True
            )
            
            # Return home
            print("Returning home...")
            robot.move_to_home(wait=True)
            
        else:
            print("Failed to initialize robot!")
            sys.exit(1)
    
    print("Done!")
