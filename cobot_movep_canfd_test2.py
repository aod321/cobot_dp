#%%
from rm_65_model import RM65
from cobot_controller import RobotController
import time

rm_65_ik_model = RM65()
cobot_controller = RobotController()

# %%
cobot_controller.connect()

# %%
def servop(target_pose, current_pose, speed=0.1, control_period=0.02):
    """
    Servo to target pose with speed control
    Args:
        target_pose: Target pose [x,y,z,rx,ry,rz] (meters, radians)
        current_pose: Current pose [x,y,z,rx,ry,rz]
        speed: Maximum speed in m/s
        control_period: Control cycle time in seconds
    Returns:
        List: Next interpolated pose
    """
    # Calculate maximum step size based on speed and period
    max_step = speed * control_period
    
    # Calculate total distance
    pos_diff = [t - c for t, c in zip(target_pose[:3], current_pose[:3])]
    distance = (sum(x*x for x in pos_diff)) ** 0.5
    
    if distance < max_step:
        # Close enough, just go to target
        next_pose = target_pose
    else:
        # Interpolate position
        scale = max_step / distance
        next_pose = [
            current_pose[i] + (target_pose[i] - current_pose[i]) * scale 
            for i in range(len(current_pose))
        ]
    
    # Send servo command
    cobot_controller.movep_canfd(next_pose, follow=False)
    return next_pose

# Get current pose
current_pose = cobot_controller.get_end_pose()

# Set target pose slightly offset from current
target_pose = current_pose.copy()
target_pose[1] -= 0.01  # Move 1cm in Y direction

t_start = time.time()

while True:
    current_pose = servop(target_pose, current_pose, speed=0.5)
    
    # Check if reached target
    if current_pose == target_pose:
        break
        
    # Wait for next cycle
    t_elapsed = time.time() - t_start
    if t_elapsed < 0.02:
        time.sleep(0.02 - t_elapsed)
    t_start = time.time()

# %%
