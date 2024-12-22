import multiprocessing as mp
from multiprocessing import Queue
import time
from dataclasses import dataclass
import numpy as np
from rm_65_model import RM65
from cobot_controller import RobotController
from spatialmath import SE3
import roboticstoolbox as rtb
import pyspacemouse
import cv2

@dataclass
class MoveCommand:
    direction: str  # 'pose', 'gripper', 'estop', 'clear'
    value: float

class SpaceMouseController(mp.Process):
    def __init__(self, command_queue: Queue):
        super().__init__()
        self.command_queue = command_queue
        # self.linear_scale = 0.001  # 线性移动缩放因子 (meters)
        self.linear_scale = 0.001  # 线性移动缩放因子 (meters)
        self.angular_scale = 0.001  # 角度移动缩放因子 (radians)
        self.gripper_state = False
        self.button_debounce_time = 0.2  # 按钮防抖时间(秒)
        self.last_button_press = 0
        self.lock_z = False  # 锁定z轴和旋转的标志
        self.deadzone = 0.0001  # 增加死区阈值变量
        
    def run(self):
        success = pyspacemouse.open()
        if not success:
            print("Failed to connect to SpaceMouse!")
            return
            
        print("SpaceMouse control started")
        print("Button 0: Toggle gripper")
        print("Button 1: Emergency stop")
        print("Press 'l' to toggle lock z-axis and rotation")
        
        # 创建一个命名窗口用于接收键盘输入
        cv2.namedWindow('Keyboard Control', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Keyboard Control', 1, 1)
        
        while True:
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('l'):
                self.lock_z = not self.lock_z
                print(f"Z-axis and rotation {'locked' if self.lock_z else 'unlocked'}")
            
            state = pyspacemouse.read()
            if state:
                # 应用死区
                x = state.x if abs(state.x) > self.deadzone else 0
                y = state.y if abs(state.y) > self.deadzone else 0
                z = state.z if abs(state.z) > self.deadzone else 0
                roll = state.roll if abs(state.roll) > self.deadzone else 0
                pitch = state.pitch if abs(state.pitch) > self.deadzone else 0
                yaw = state.yaw if abs(state.yaw) > self.deadzone else 0
                
                # 缩放输入值
                if self.lock_z:
                    # 锁定z轴和旋转时，只保留x和y轴的移动
                    pose_delta = [
                        -y * self.linear_scale,   # x (原来的y)
                        x * self.linear_scale,   # y (原来的x)
                        0,   # z locked
                        0,   # rx locked
                        0,   # ry locked
                        0    # rz locked
                    ]
                else:
                    pose_delta = [
                        -y * self.linear_scale,   # x (原来的y)
                        x * self.linear_scale,   # y (原来的x)
                        z * self.linear_scale,   # z
                        roll * self.angular_scale,  # rx
                        pitch * self.angular_scale, # ry
                        yaw * self.angular_scale    # rz
                    ]
                
                # 检查是否有显著运动
                threshold = 0.00001
                if any(abs(v) > threshold for v in pose_delta):
                    self.command_queue.put(MoveCommand('pose', pose_delta))
                
                # 处理按钮输入（带防抖）
                current_time = time.time()
                if current_time - self.last_button_press > self.button_debounce_time:
                    if state.buttons[0]:  # 第一个按钮控制夹持器
                        self.gripper_state = not self.gripper_state
                        self.command_queue.put(MoveCommand('gripper', float(self.gripper_state)))
                        self.last_button_press = current_time
                    elif state.buttons[1]:  # 第二个按钮复位
                        # self.cobot_controller.emergency_stop()
                        # self.cobot_controller.update_current_pose()
                        # self.command_queue.put(MoveCommand('estop', 0))
                        self.command_queue.put(MoveCommand('clear', 0))
                        self.last_button_press = current_time
                    
            time.sleep(0.01)

class RobotControlProcess(mp.Process):
    def __init__(self, command_queue: Queue):
        super().__init__()
        self.command_queue = command_queue
        self.current_pose = None  # Will be initialized in run()
        self.current_joints = None  # Will be initialized in run()
        self.current_joints_rad = None  # Will be initialized in run()

    def servoL(self, target_pose, target_pose_as_next_current_pose=False, use_movejp=False):
        # 分别获取当前和目标的位置与��态
        current_pos = self.current_pose[:3]
        current_rpy = self.current_pose[3:]
        target_pos = target_pose[:3]
        target_rpy = target_pose[3:]

        # 位置插值
        via_points = np.array([current_pos, target_pos])
        pos_traj = rtb.mstraj(
            via_points,     # via points
            # dt=0.02,        # time step (s)
            dt=0.01,        # time step (s)
            # tacc=0.01,      # acceleration time (s)
            tacc=0.03,      # acceleration time (s)
            qdmax=1.5,        # max velocity (m/s)
        )

        # 姿态插值 - 使用Slerp
        num_steps = len(pos_traj.q)
        R0 = SE3.RPY(current_rpy)
        R1 = SE3.RPY(target_rpy)
        rpy_traj = np.zeros((num_steps, 3))
        
        for i in range(num_steps):
            s = i / (num_steps - 1)  # 插值参数，从0到1
            R_interp = R0.interp(R1, s)  # Slerp插值
            rpy_traj[i] = R_interp.rpy()

        self.cobot_controller.speed_upbound = 50
        q_prev = self.current_joints_rad
        rad2deg = 180 / np.pi
        if use_movejp:
            # 执行轨迹
            for pos, rpy in zip(pos_traj.q, rpy_traj):
                T = SE3(pos) * SE3.RPY(rpy)
                q = self.cobot_controller.ik_model.ik_LM(T,q0=q_prev)  # Changed pos_traj to pos
                target_joints_angle = (q[0] * rad2deg).tolist()
                self.cobot_controller.movej_canfd(target_joints_angle, follow=False)
                q_prev = q[0]
        else:
            for pos, rpy in zip(pos_traj.q, rpy_traj):
                to_pose = pos.tolist() + rpy.tolist()
                self.cobot_controller.movep_canfd(to_pose, follow=False)
        if target_pose_as_next_current_pose:
            self.current_pose = target_pose
        else:
            self.update_current_pose()

    def update_current_pose(self):
        deg2rad = np.pi/180
        self.current_pose = self.cobot_controller.calc_end_pose()
        self.current_joints = self.cobot_controller.get_joint_angles()
        self.current_joints_rad = np.array(self.current_joints) * deg2rad
        
    def run(self):
        self.cobot_controller = RobotController()
        self.rm_65_ik_model = RM65()
        self.cobot_controller.connect()
        self.cobot_controller.speed_upbound = 50
        self.update_current_pose()
        
        print("机械臂控制已启动")
        
        while True:
            try:
                command = self.command_queue.get_nowait()
                if command.direction == 'estop':
                    self.cobot_controller.emergency_stop()
                elif command.direction == 'clear':
                    self.cobot_controller.clear_error()
                    self.cobot_controller.resume()
                    self.update_current_pose()
                elif command.direction == 'gripper':
                    if command.value == 1:
                        self.cobot_controller.gripper.open()
                        command.value = -1
                    elif command.value == 0:
                        self.cobot_controller.gripper.close()
                        command.value = -1
                elif command.direction == 'pose':
                    # command.value 包含所有轴的增量 [dx, dy, dz, drx, dry, drz]
                    new_pose = self.current_pose.copy()
                    for i in range(6):
                        new_pose[i] += command.value[i]
                    self.servoL(new_pose, target_pose_as_next_current_pose=True, use_movejp=False)
                
            except Exception as e:
                # print(f"Error: {e}")
                time.sleep(0.01)  # 没有新命令时短暂休眠

def main():
    command_queue = Queue()
    
    # 创建并启动控制进程
    spacemouse_controller = SpaceMouseController(command_queue)
    robot_controller = RobotControlProcess(command_queue)
    
    spacemouse_controller.start()
    robot_controller.start()
    
    # 等待进程结束
    spacemouse_controller.join()
    robot_controller.terminate()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()