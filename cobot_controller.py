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
            
    def send_command(self, command):
        """Send command to robot and receive response"""
        if not self.socket:
            print("ERROR: Not connected to robot")
            return None
            
        try:
            # Add newline to command as required by protocol
            command = command + "\r\n"
            self.socket.send(command.encode())
            
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
