#%%
import socket
import json
import time
import select
from gripper_controller import GripperController
from rm_65_model import RM65
import numpy as np

class RobotController:
    def __init__(self, ip="192.168.40.102", port=8080,
                 gripper_ip="192.168.40.102", gripper_port=8000,
                 speed_upbound=5,
                 home_expand_joint=-90, home_joint1_6=[-15.805, 1.628, -104.147, 6.648, -70.063, 20.478]):
        """Initialize robot controller with IP and port"""
        self.ip = ip
        self.port = port
        self.socket = None
        self.home_expand_joint = home_expand_joint
        self.home_joint1_6 = home_joint1_6
        self.gripper = GripperController(base_url=f"http://{gripper_ip}:{gripper_port}")
        self.speed_upbound = speed_upbound
        self.ik_model = RM65()
    
    def connect(self):
        """Connect to the robot"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.ip, self.port))
            print("INFO: Successfully connected to robot")
            return True
        except Exception as e:
            print(f"ERROR: Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect safely from robot"""
        try:
            # First stop any ongoing movement
            stop_command = {"command": "set_arm_stop"}
            self.send_command(json.dumps(stop_command))
            
            # Close socket if it exists
            if self.socket:
                self.socket.close()
                self.socket = None
                print("INFO: Successfully disconnected from robot")
                return True
                
        except Exception as e:
            print(f"ERROR: Error during disconnect: {e}")
        
        self.socket = None
        return False
            
    def send_command(self, command, wait_response=True):
        """Send command to robot and receive response"""
        if not self.socket:
            print("ERROR: Not connected to robot")
            return None
            
        try:
            # Add newline to command as required by protocol
            command = command + "\r\n"
            self.socket.send(command.encode())
            
            if wait_response:
                # Use select to wait for response with timeout
                ready = select.select([self.socket], [], [], 2.0)
                if ready[0]:
                    response = self.socket.recv(1024).decode()
                    # Clear any remaining data in buffer
                    while select.select([self.socket], [], [], 0.1)[0]:
                        self.socket.recv(1024)
                    return json.loads(response)
                else:
                    print("ERROR: Timeout waiting for response")
                    return None
            return True
                
        except Exception as e:
            print(f"ERROR: Error sending command: {e}")
            return None

    def calc_end_pose(self):
        """
        This function exists because the get_current_arm_state interface provided by the robot manufacturer
        is very unstable and frequently returns errors.
        Calculate end effector pose from current joint angles
        """
        deg2rad = np.pi/180
        current_joint_angles = self.get_joint_angles()  
        current_joints_rad = np.array(current_joint_angles) * deg2rad
        T0 = self.ik_model.fkine(current_joints_rad)
        return T0.t.tolist() + T0.rpy().tolist()
    
    def get_end_pose(self):
        """
        Get current end effector pose
        Returns: List of [x, y, z, rx, ry, rz] where:
            x,y,z: Position in meters
            rx,ry,rz: Orientation in radians
            Returns None if failed
        """
        command = {
            "command": "get_current_arm_state"
        }
        response = self.send_command(json.dumps(command))
        
        if response and "arm_state" in response:
            pose_units = response["arm_state"]["pose"]
            # Convert position from protocol units (0.001mm) to meters
            pose = [pos/1000000.0 for pos in pose_units[:3]]
            # Convert orientation from protocol units (0.001rad) to radians
            pose.extend([angle/1000.0 for angle in pose_units[3:]])
            return pose
        else:
            print("ERROR: Failed to get end effector pose")
            return None

    def get_expand_position(self):
        """Get current expand axis position in degrees"""
        command = {
            "command": "expand_get_state"
        }
        response = self.send_command(json.dumps(command))
        if response and "state" in response:
            if response.get('pos') != 'unknown':
                pos_deg = response.get('pos') / 1000.0  # Convert to degrees
                print(f"Current expand position: {pos_deg} degrees")
                print(f"Current mode: {response.get('mode', 'unknown')}")
                print(f"Error flag: {response.get('err_flag', 'unknown')}")
                return pos_deg
        print("ERROR: Failed to get expand position")
        return None

    def get_joint_angles(self):
        """
        Get current angles of joints 1-6 (not including expand axis)
        Returns: List of 6 joint angles in degrees or None if failed
        """
        command = {
            "command": "get_joint_degree"
        }
        response = None
        while response is None or response.get('state', '') != 'joint_degree': # retry
            response = self.send_command(json.dumps(command))
        if response and "state" in response:
            # Convert from protocol units (0.001 degrees) to degrees
            joint_angles = [angle/1000.0 for angle in response["joint"]]
            return joint_angles
        else:
            print("ERROR: Failed to get joint angles")
            return None
            
    def move_expand_axis(self, target_pos, speed=5):
        """
        Move expand axis to target position
        Args:
            target_pos: Target position in degrees
            speed: Movement speed (1-100)
        Returns:
            bool: True if movement successful, False otherwise
        """
        if not 1 <= speed <= 100:
            print("ERROR: speed must be between 1 and 100")
            return False
            
        speed = self.speed_upbound if speed >= self.speed_upbound else speed
        
        # Convert to protocol units (0.001 degrees)
        pos_units = int(target_pos * 1000)
        
        expand_command = {
            "command": "expand_set_pos",
            "pos": pos_units,
            "speed": speed
        }
        
        return self._execute_expand_movement(expand_command)
    
    def moveL(self, target_pose, speed=5, in_trajectory=False):
        """
        Move robot end effector in a straight line to target pose
        Args: 
            target_pose: List of [x, y, z, rx, ry, rz] where:
                x,y,z: Target position in meters 
                rx,ry,rz: Target orientation in radians
            speed: Movement speed (1-100)
        Returns:
            bool: True if movement successful, False otherwise
        """
        if not 1 <= speed <= 100:
            print("ERROR: speed must be between 1 and 100")
            return False
        
        speed = self.speed_upbound if speed >= self.speed_upbound else speed
        
        # Convert to protocol units (0.001 mm for position, 0.001 rad for orientation)
        pose_units = []
        # Position conversion from meters to millimeters (*1000) then to protocol units (*1000)
        for i in range(3):
            pose_units.append(int(target_pose[i] * 1000000))
        # Orientation conversion from radians to protocol units (*1000) 
        for i in range(3,6):
            pose_units.append(int(target_pose[i] * 1000))

        movel_command = {
            "command": "movel",
            "pose": pose_units,
            "v": speed,
            "r": 0,
            "trajectory_connect": 1 if in_trajectory else 0
        }
        
        response = self.send_command(json.dumps(movel_command))
        if response and response.get("receive_state"):
            print(f"INFO: Successfully commanded move to target pose")
            return True
            
        print("ERROR: Failed to send movement command") 
        return False
    
    def movej_canfd(self, joint_angles, follow=False, expand_angle=None):
        """
        Joint angle servo without trajectory planning
        Args:
            joint_angles: List of 6 angles in degrees for joints 1-6
            follow: True for high following mode (requires cycle time ≤10ms)
                False for low following mode
            expand_angle: Optional expand axis angle in degrees
        Returns:
            bool: True if command sent successfully, False otherwise
        Notes:
            - Requires stable control cycle: 20ms for WiFi/network, 10ms for USB/RS485/high-speed port, 2ms for I series
            - Joint angle change between frames must be <10° and speed <180°/s
            - No trajectory planning, direct joint control
        """
        if not isinstance(joint_angles, list) or len(joint_angles) != 6:
            print("ERROR: joint_angles must be a list of 6 angles")
            return False
            
        # Get current angles and check angle changes
        current_angles = self.get_joint_angles()
        if current_angles:
            for curr, target in zip(current_angles, joint_angles):
                if abs(target - curr) > 10:
                    print("ERROR: Joint angle change too large (>10°)")
                    return False
        
        # Convert angles to protocol units (0.001 degrees)
        joint_units = [int(angle * 1000) for angle in joint_angles]
        
        command = {
            "command": "movej_canfd",
            "joint": joint_units,
            "follow": follow
        }
        
        # Add expand axis if provided
        if expand_angle is not None:
            command["expand"] = int(expand_angle * 1000)
            
        response = self.send_command(json.dumps(command), wait_response=False)
        if response:
            print(f"INFO: Successfully sent joint servo command")
            return True
                
        print("ERROR: Failed to send joint servo command") 
        return False

    def movep_canfd(self, target_pose, follow=False, use_quaternion=False):
        """
        Pose servo without trajectory planning
        Args:
            target_pose: Two format options:
                1. List of [x, y, z, rx, ry, rz] where:
                    x,y,z: Target position in meters
                    rx,ry,rz: Target orientation in radians
                2. List of [x, y, z, qw, qx, qy, qz] where:
                    x,y,z: Target position in meters
                    qw,qx,qy,qz: Orientation in quaternion
            follow: True for high following mode (requires cycle time ≤10ms)
                    False for low following mode
            use_quaternion: True to use quaternion format, False to use euler angles
        Returns:
            bool: True if command sent successfully, False otherwise
        Notes:
            - Requires stable control cycle: 
            > 20ms for WiFi/network
            > 10ms for USB/RS485/high-speed port
            > 2ms for I series
            - No trajectory planning, direct inverse kinematics
        """
        if use_quaternion and len(target_pose) != 7:
            print("ERROR: Quaternion pose must be [x,y,z,qw,qx,qy,qz]")
            return False
        elif not use_quaternion and len(target_pose) != 6:
            print("ERROR: Euler pose must be [x,y,z,rx,ry,rz]")
            return False

        pose_units = []
        # Position conversion from meters to millimeters (*1000) then to protocol units (*1000)
        for i in range(3):
            pose_units.append(int(target_pose[i] * 1000000))
            
        # Orientation conversion depends on format
        if use_quaternion:
            # Convert quaternion to protocol units (*1000000)
            for i in range(3,7):
                pose_units.append(int(target_pose[i] * 1000000))
            command = {
                "command": "movep_canfd",
                "pose_quat": pose_units,
                "follow": follow
            }
        else:
            # Convert euler angles to protocol units (*1000)
            for i in range(3,6):
                pose_units.append(int(target_pose[i] * 1000))
            command = {
                "command": "movep_canfd",
                "pose": pose_units,
                "follow": follow
            }
            
        response = self.send_command(json.dumps(command), wait_response=False)
        if response:
            return True
                
        print("ERROR: Failed to send pose servo command") 
        return False
    
    def emergency_stop(self):
        """
        Stop robot immediately with maximum deceleration.
        Trajectory cannot be resumed after emergency stop.
        Returns:
            bool: True if command successful, False otherwise
        """
        command = {
            "command": "set_arm_stop"
        }
        response = self.send_command(json.dumps(command))
        if response and response.get("arm_stop"):
            print("INFO: Emergency stop executed")
            return True
        print("ERROR: Failed to execute emergency stop")
        return False

    def clear_error(self):
        """
        Clear system error state to allow new commands
        Returns:
            bool: True if successful, False otherwise
        """
        command = {
            "command": "clear_system_err"
        }
        response = self.send_command(json.dumps(command))
        if response and response.get("clear_state"):
            print("INFO: System error cleared")
            return True
        print("ERROR: Failed to clear system error")
        return False

    def slow_stop(self):
        """
        Stop robot smoothly on current trajectory.
        Returns:
            bool: True if command successful, False otherwise
        """
        command = {
            "command": "set_arm_slow_stop"
        }
        response = self.send_command(json.dumps(command))
        if response and response.get("arm_slow_stop"):
            print("INFO: Slow stop executed")
            return True
        print("ERROR: Failed to execute slow stop")
        return False

    def pause(self):
        """
        Pause robot movement. Movement can be resumed later.
        Returns:
            bool: True if command successful, False otherwise
        """
        command = {
            "command": "set_arm_pause"
        }
        response = self.send_command(json.dumps(command))
        if response and response.get("arm_pause"):
            print("INFO: Movement paused")
            return True
        print("ERROR: Failed to pause movement")
        return False

    def resume(self):
        """
        Resume paused movement.
        Returns:
            bool: True if command successful, False otherwise
        """
        command = {
            "command": "set_arm_continue"
        }
        response = self.send_command(json.dumps(command))
        if response and response.get("arm_continue"):
            print("INFO: Movement resumed")
            return True
        print("ERROR: Failed to resume movement")
        return False

    def clear_trajectory(self):
        """
        Clear current trajectory. Must be called in paused state.
        Returns:
            bool: True if command successful, False otherwise
        """
        command = {
            "command": "set_delete_current_trajectory"
        }
        response = self.send_command(json.dumps(command))
        if response and response.get("delete_current_trajectory"):
            print("INFO: Current trajectory cleared")
            return True
        print("ERROR: Failed to clear trajectory")
        return False

    def clear_all_trajectories(self):
        """
        Clear all trajectories. Must be called in paused state.
        Returns:
            bool: True if command successful, False otherwise
        """
        command = {
            "command": "set_arm_delete_trajectory"
        }
        response = self.send_command(json.dumps(command))
        if response and response.get("arm_delete_trajectory"):
            print("INFO: All trajectories cleared")
            return True
        print("ERROR: Failed to clear trajectories")
        return False
    
    def _parse_error_code(self, err_code):
        """Parse system error code to human readable text"""
        ERROR_CODES = {
            0x0000: "正常",
            0x1001: "关节通信异常",
            0x1002: "目标角度超过限位",
            0x1003: "该处不可达，为奇异点", 
            0x1004: "实时内核通信错误",
            0x1007: "关节超速",
            0x1008: "末端接口板无法连接",
            0x1009: "超速度限制",
            0x100A: "超加速度限制",
            0x100B: "关节抱闸未打开",
            0x100C: "拖动示教时超速",
            0x100D: "机械臂发生碰撞",
            0x100E: "无该工作坐标系",
            0x100F: "无该工具坐标系",
            0x1010: "关节发生掉使能错误",
            0x1011: "圆弧规划错误",
            0x1012: "自碰撞错误",
            0x1013: "碰撞到电子围栏错误",
            0x1014: "超关节软限位错误"
        }
        return ERROR_CODES.get(err_code, f"未知错误码: {hex(err_code)}")

    def get_system_state(self):
        """
        Get current system state including voltage, current, temperature and error flags
        Returns:
            dict: System state info if successful, None if failed
        """
        command = {
            "command": "get_controller_state"
        }
        response = self.send_command(json.dumps(command))
        
        if response and "state" in response:
            # Convert from protocol units (0.001) to actual values
            state = {
                'voltage': response.get('voltage', 0) / 1000.0,  # V
                'current': response.get('current', 0) / 1000.0,  # A
                'temperature': response.get('temperature', 0) / 1000.0,  # °C
                'err_flag': response.get('err_flag', 0),  # Error code
                'err_msg': self._parse_error_code(response.get('err_flag', 0))
            }
            print(f"INFO: System state:")
            print(f"- Voltage: {state['voltage']}V")
            print(f"- Current: {state['current']}A")
            print(f"- Temperature: {state['temperature']}°C")
            print(f"- Error: {state['err_msg']} ({hex(state['err_flag'])})")
            return state
        
        print("ERROR: Failed to get system state")
        return None

    def move_joints(self, joint_angles, speed=5):
        """
        Move robot joints 1-6 to specified angles
        Args:
            joint_angles: List of 6 angles in degrees for joints 1-6
            speed: Movement speed (1-100)
        Returns:
            bool: True if movement successful, False otherwise
        """
        if not isinstance(joint_angles, list) or len(joint_angles) != 6:
            print("ERROR: joint_angles must be a list of 6 angles")
            return False
            
        if not 1 <= speed <= 100:
            print("ERROR: speed must be between 1 and 100")
            return False
        
        speed = self.speed_upbound if speed >= self.speed_upbound else speed

        print(f"INFO: Moving joints to angles: {joint_angles} at speed {speed}")
        
        # Convert angles to protocol units (0.001 degrees)
        joint_units = [int(angle * 1000) for angle in joint_angles]

        joint_command = {
            "command": "movej",
            "joint": joint_units,
            "v": speed,
            "r": 0,
            "trajectory_connect": 0
        }
        
        response = self.send_command(json.dumps(joint_command))
        if response and response.get("receive_state"):
            print(f"INFO: Successfully commanded joints to move to {joint_angles} degrees")
            return True
        
        print("ERROR: Failed to send joint movement command")
        return False
    
    def move_single_joint(self, joint_index, target_angle, speed=5):
        """
        Move a single joint to target angle while keeping other joints static
        Args:
            joint_index: Joint number (1-6)
            target_angle: Target angle in degrees
            speed: Movement speed (1-100)
        Returns:
            bool: True if movement successful, False otherwise
        """
        if joint_index == 0:
            return self.move_expand_axis(target_angle, speed)

        if not 1 <= joint_index <= 6:
            print("ERROR: joint_index must be between 1 and 6")
            return False

        # Get current joint positions
        current_angles = self.get_joint_angles()
        if current_angles is None:
            print("ERROR: Failed to get current joint angles")
            return False
            
        joint_angles = current_angles.copy()
        joint_angles[joint_index-1] = target_angle

        return self.move_joints(joint_angles, speed)

    def move_joint_by_delta(self, joint_index, delta_angle, speed=5):
        """
        Move a single joint by a delta angle relative to current position
        Args:
            joint_index: Joint number (1-6) or 'expand' for expand axis
            delta_angle: Change in angle in degrees (positive or negative)
            speed: Movement speed (1-100)
        Returns:
            bool: True if movement successful, False otherwise
        """
        time.sleep(0.1)  # Add delay before getting positions
        
        if joint_index == 'expand' or joint_index == 0:
            current_pos = self.get_expand_position()
            if current_pos is None:
                return False
            target_pos = current_pos + delta_angle
            return self.move_expand_axis(target_pos, speed)
        else:
            if not 1 <= joint_index <= 6:
                print("ERROR: joint_index must be between 1 and 6")
                return False
                
            # Get current joint positions
            current_angles = self.get_joint_angles()
            if current_angles is None:
                print("ERROR: Failed to get current joint angles")
                return False
                
            # Calculate target angle by adding delta to current angle
            target_angle = current_angles[joint_index-1] + delta_angle
            return self.move_single_joint(joint_index, target_angle, speed)

    def _execute_expand_movement(self, command):
        """
        Execute expand axis movement and wait for completion
        Args:
            command: Expand axis movement command dictionary
        Returns:
            bool: True if movement successful, False otherwise
        """
        response = self.send_command(json.dumps(command))
        if not response or not response.get("set_pos_state"):
            print("ERROR: Failed to send expand axis command")
            return False
            
        print("INFO: Expand axis command accepted")
        # Wait for movement completion with timeout
        try:
            timeout = time.time() + 10  # 10 second timeout
            while time.time() < timeout:
                ready = select.select([self.socket], [], [], 1.0)
                if ready[0]:
                    response = self.socket.recv(1024).decode()
                    arrival = json.loads(response)
                    if (arrival.get("state") == "current_trajectory_state" and 
                        arrival.get("device") == 4 and 
                        arrival.get("trajectory_state")):
                        print("INFO: Expand axis reached target position")
                        return True
                time.sleep(0.1)  # Add small delay in loop
            print("ERROR: Timeout waiting for expand axis movement")
            return False
        except (json.JSONDecodeError, socket.error) as e:
            print(f"ERROR: Failed while waiting for expand axis movement: {e}")
            return False
    
    def go_home_pos(self):
        """Move to home position"""
        self.move_expand_axis(self.home_expand_joint, speed=self.speed_upbound)
        self.move_joints(self.home_joint1_6, speed=self.speed_upbound)
# %%
if __name__ == "__main__":
    import numpy as np
    robot = RobotController()
    robot.connect()
    joints_angles = robot.get_joint_angles()
    deg2rad = np.pi/180
    joints_rad = np.array(joints_angles) * deg2rad  # Convert to numpy array before multiplying
    current_end_pose = robot.get_end_pose()
    from rm_65_model import RM65
    model = RM65()
    current_end_pose_model = model.fkine(joints_rad)
    # Convert SE3 object to list of position and orientation
    current_end_pose_model_list = current_end_pose_model.t.tolist() + current_end_pose_model.rpy().tolist()
    print(f"Joints angles: {joints_angles}")
    print(f"Current end pose: {current_end_pose}")
    print(f"Current end pose model: {current_end_pose_model_list}")
