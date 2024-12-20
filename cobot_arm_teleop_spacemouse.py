#%%
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from cobot_controller import RobotController
from mini_ik import create_rm65b_robot
class CobotArmTeleopSpacemouse:
    def __init__(self):
        push_block_init_joint1_6_pos = [-15.805, 1.628, -104.147, 6.648, -70.063, 20.478]
        push_block_init_expand_pos = -90
        self.robot = RobotController(home_expand_joint=push_block_init_expand_pos,
                                    home_joint1_6=push_block_init_joint1_6_pos)
        self.robot.connect()
        self.cobot_model = create_rm65b_robot()
        self.ik_result = None
        # self.reset_robot()
    
    def reset_robot(self):
        self.robot.go_home_pos()
        self.robot.gripper.close()

    def run(self):
        pass
    
    def calculate_eef_to_target_pos(self, delta_list, visualize=False):
        """
        Move end-effector by specified deltas in position and orientation
        Args:
            delta_list: List of 6 values [dx, dy, dz, drx, dry, drz]
                dx, dy, dz: Translation in x,y,z direction (centimeters)
                drx, dry, drz: Rotation around x,y,z axis (degrees)
        Returns:
            bool: True if movement successful, False otherwise
        """
        deg2rad = np.pi / 180
        rad2deg = 180.0 / np.pi
        # Get current joint angles
        current_angles = self.robot.get_joint_angles()
        if current_angles is None:
            print("ERROR: Failed to get current joint angles")
            return False
            
        current_angles = np.array(current_angles) * deg2rad
        # Get current pose
        current_pose = self.cobot_model.fkine(current_angles)
        if visualize:
            # visual current pose
            self.cobot_model.plot(current_angles)
        
        # Create delta transform - convert cm to meters for SE3
        delta_pose = SE3.Tx(delta_list[0]/100) * SE3.Ty(delta_list[1]/100) * SE3.Tz(delta_list[2]/100) * \
                    SE3.Rx(delta_list[3] * deg2rad) * SE3.Ry(delta_list[4] * deg2rad) * SE3.Rz(delta_list[5] * deg2rad)
        
        # Calculate target pose
        target_pose = current_pose * delta_pose
        
        # Solve inverse kinematics
        sol = self.cobot_model.ik_LM(target_pose)
        if sol[0] is None:
            print("ERROR: Failed to solve inverse kinematics")
            return False
            
        # Convert target angles from radians to degrees
        target_angles = sol[0] * rad2deg

        # Visualize the IK result in gif
        qt = rtb.jtraj(current_angles, sol[0], 50)
        if visualize:
            self.cobot_model.plot(qt.q, backend='pyplot', movie='cobot_ik.gif')
        self.ik_result = target_angles
        return target_angles

    def run_ik(self):
        if self.ik_result is None:
            print("ERROR: No IK result found")
            return False
        self.robot.move_joints(list(self.ik_result))
        return True

    def calculate_eef_to_target_pos_min_rotation(self, delta_xyz, visualize=False, gif_name='cobot_ik.gif',
                                                weights=None):
        """
        Move end-effector to target position (x,y,z) while minimizing rotation changes
        Args:
            delta_xyz: List of 3 values [dx, dy, dz] for translation in x,y,z direction (centimeters)
        Returns:
            target_angles if successful, False otherwise
        """
        deg2rad = np.pi / 180
        rad2deg = 180.0 / np.pi
        
        # Get current joint angles
        current_angles = self.robot.get_joint_angles()
        if current_angles is None:
            print("ERROR: Failed to get current joint angles")
            return False
            
        current_angles = np.array(current_angles) * deg2rad
        # Get current pose
        current_pose = self.cobot_model.fkine(current_angles)
        if visualize:
            self.cobot_model.plot(current_angles)
        
        # Create delta transform - only for position
        delta_pose = SE3.Tx(delta_xyz[0]/100) * SE3.Ty(delta_xyz[1]/100) * SE3.Tz(delta_xyz[2]/100)
        
        # Calculate target pose
        target_pose = current_pose * delta_pose
        
        # Solve IK with mask to prioritize position matching over orientation
        if weights is None:
            weights = np.array([1,1,1,0.5,0.5,0.5])
        
        sol = self.cobot_model.ik_LM(Tep=target_pose, mask=weights,ilimit=50, slimit=150)
        if sol[0] is None:
            print("ERROR: Failed to solve inverse kinematics")
            return False
            
        # Convert target angles from radians to degrees
        target_angles = sol[0] * rad2deg

        if visualize:
            qt = rtb.jtraj(current_angles, sol[0], 50)
            self.cobot_model.plot(qt.q, backend='pyplot', movie=gif_name)
            
        self.ik_result = target_angles
        return target_angles

def grid_search_rotation_weights():
    import os
    import numpy as np

    # Create output folder if it doesn't exist
        
    # Test range from -5 to 5 with step size 0.1
    # test_range = np.arange(-5, 5, 0.1)
    test_range = [0.5]

    weights = np.array([1,1,1,0.2,0.2,0.2])
    # grid search for rotation weights
    rotation_weights =np.arange(0.1,0.25,0.05)
    for weight in rotation_weights:
        output_folder = f"ik_test_results/weight_{weight}"
        weights[3:] = weight
        os.makedirs(output_folder, exist_ok=True)
        # Test movements along x axis
        for x in test_range:
            gif_name = os.path.join(output_folder, f'cobot_ik_x_{x}.gif')
            cobot.calculate_eef_to_target_pos_min_rotation([x, 0, 0], visualize=True, gif_name=gif_name, weights=weights)
            
        # Test movements along y axis  
        for y in test_range:
            gif_name = os.path.join(output_folder, f'cobot_ik_y_{y}.gif')
            cobot.calculate_eef_to_target_pos_min_rotation([0, y, 0], visualize=True, gif_name=gif_name, weights=weights)
            
        # Test movements along z axis
        for z in test_range:
            gif_name = os.path.join(output_folder, f'cobot_ik_z_{z}.gif')
            cobot.calculate_eef_to_target_pos_min_rotation([0, 0, z], visualize=True, gif_name=gif_name, weights=weights)

#%%
if __name__ == "__main__":
    cobot = CobotArmTeleopSpacemouse()
    # cobot.reset_robot(
    # cobot.calculate_eef_to_target_pos([0, 0, 0, 0, 0, 0], visualize=True)

# %%
