import multiprocessing as mp
from multiprocessing import Queue
import cv2
import time
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from rm_65_model import RM65
from cobot_controller import RobotController
from spatialmath import SE3
import roboticstoolbox as rtb

@dataclass
class MoveCommand:
    direction: str  # 'x', 'y', 'z', 'gripper', 'clear'
    value: float
    
class KeyboardController(mp.Process):
    def __init__(self, command_queue: Queue):
        super().__init__()
        self.command_queue = command_queue
        self.step_size = 0.01  # 每次移动的步长(米)
        self.rotation_step = 5  # 每次旋转的步长(角度)
        self.rotation_step_rad = self.rotation_step * np.pi / 180 # 旋转步长(弧度)
        self.gripper_state = False  # False为关闭，True为打开 
        # 添加按键状态追踪
        self.pressed_keys = set()
        
    def run(self):
        # 创建控制说明文本
        control_instructions = [
            "Keyboard Control Instructions:",
            "X axis: A/D",
            "Y axis: W/S", 
            "Z axis: Q/E",
            "End joint rotation: R/F",
            "Gripper: Space",
            "Stop: P",
            "E-Stop: O", 
            "Clear error: C",
            "System status: I",
            "Exit: ESC",
            "End rotation control:",
            "Around X: 1(+)/2(-)",
            "Around Y: 3(+)/4(-)",
            "Around Z: 5(+)/6(-)"
        ]

        # 打印到控制台（这里可以保持中文）
        print("键盘控制已启动")
        for instruction in control_instructions:
            print(instruction)
        
        # 创建窗口
        cv2.namedWindow('Robot Control', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Robot Control', 400, 600)
        
        # 创建显示控制说明的图像
        img = np.zeros((600, 400, 3), dtype=np.uint8)
        
        # 在图像上添加文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 1
        line_spacing = 30
        
        for i, text in enumerate(control_instructions):
            y = 30 + i * line_spacing
            cv2.putText(img, text, (10, y), font, font_scale, (255, 255, 255), font_thickness)
        
        cv2.imshow('Robot Control', img)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # 检测按键释放
            if key == 255:  # 无按键按下
                if self.pressed_keys:  # 如果之前有按键被按下
                    # 清空命令队列
                    while not self.command_queue.empty():
                        try:
                            self.command_queue.get_nowait()
                        except:
                            break
                    self.pressed_keys.clear()
                continue

            # 记录按下的按键
            self.pressed_keys.add(key)
            
            # 添加调试信息
            if key != 255:  # 255是无按键时的默认值
                print(f"Pressed key code: {key}")
            
            if key == ord('p') or key == ord('o'):
                # 清空命令队列
                while not self.command_queue.empty():
                    try:
                        self.command_queue.get_nowait()
                    except:
                        break
                        
                # 插入急停或停止命令
                if key == ord('p'):
                    self.command_queue.put(MoveCommand('stop', 0))
                else:
                    self.command_queue.put(MoveCommand('estop', 0))
                    
            elif key == ord('a'):
                self.command_queue.put(MoveCommand('x', -self.step_size))
            elif key == ord('d'):
                self.command_queue.put(MoveCommand('x', self.step_size))
            elif key == ord('w'):
                self.command_queue.put(MoveCommand('y', self.step_size))
            elif key == ord('s'):
                self.command_queue.put(MoveCommand('y', -self.step_size))
            elif key == ord('q'):
                self.command_queue.put(MoveCommand('z', self.step_size))
            elif key == ord('e'):
                self.command_queue.put(MoveCommand('z', -self.step_size))
            elif key == 32:  # 空格键
                self.gripper_state = not self.gripper_state
                self.command_queue.put(MoveCommand('gripper', float(self.gripper_state)))
            elif key == ord('r'):
                # self.command_queue.put(MoveCommand('rz', self.rotation_step))
                self.command_queue.put(MoveCommand('j6', self.rotation_step))
            elif key == ord('f'):
                # self.command_queue.put(MoveCommand('rz', -self.rotation_step))
                self.command_queue.put(MoveCommand('j6', -self.rotation_step))
            elif key == ord('c'):
                self.command_queue.put(MoveCommand('clear', 0))
            elif key == ord('i'):
                self.command_queue.put(MoveCommand('status', 0))
            elif key == 27:  # ESC键
                break
            elif key == ord('1'):
                self.command_queue.put(MoveCommand('rx', self.rotation_step_rad))
            elif key == ord('2'):
                self.command_queue.put(MoveCommand('rx', -self.rotation_step_rad))
            elif key == ord('3'):
                self.command_queue.put(MoveCommand('ry', self.rotation_step_rad))
            elif key == ord('4'):
                self.command_queue.put(MoveCommand('ry', -self.rotation_step_rad))
            elif key == ord('5'):
                self.command_queue.put(MoveCommand('rz', self.rotation_step_rad))
            elif key == ord('6'):
                self.command_queue.put(MoveCommand('rz', -self.rotation_step_rad))
            
            # 忽略其他按键
            elif key != 255:  # 不是无按键的状态
                continue
                
        cv2.destroyAllWindows()

class RobotControlProcess(mp.Process):
    def __init__(self, command_queue: Queue):
        super().__init__()
        self.command_queue = command_queue
        self.current_pose = None  # Will be initialized in run()
        self.current_joints = None  # Will be initialized in run()
        self.current_joints_rad = None  # Will be initialized in run()

    def servoL(self, target_pose, target_pose_as_next_current_pose=False, use_movejp=False):
        # 分别获取当前和目标的位置与姿态
        current_pos = self.current_pose[:3]
        current_rpy = self.current_pose[3:]
        target_pos = target_pose[:3]
        target_rpy = target_pose[3:]

        # 位置插值
        via_points = np.array([current_pos, target_pos])
        pos_traj = rtb.mstraj(
            via_points,     # via points
            dt=0.02,        # time step (s)
            tacc=0.05,      # acceleration time (s)
            qdmax=1,        # max velocity (m/s)
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
                if command.direction == 'stop':
                    self.cobot_controller.slow_stop()
                elif command.direction == 'estop':
                    self.cobot_controller.emergency_stop()
                elif command.direction == 'clear':
                    self.cobot_controller.clear_error()
                    self.cobot_controller.resume()
                    self.update_current_pose()
                elif command.direction == 'status':
                    self.cobot_controller.get_system_state()
                elif command.direction == 'gripper':
                    if command.value == 1:
                        self.cobot_controller.gripper.open()
                        command.value = -1
                    elif command.value == 0:
                        self.cobot_controller.gripper.close()
                        command.value = -1
                elif command.direction == 'j6':
                    # print(f"关节6旋转: {command.value}")
                    current_joints = self.cobot_controller.get_joint_angles()
                    current_joints[5] += command.value
                    self.cobot_controller.movej_canfd(current_joints, follow=False)
                    self.update_current_pose()
                else:
                    new_pose = self.current_pose.copy()
                    if command.direction == 'x':
                        new_pose[1] += command.value
                    elif command.direction == 'y':
                        new_pose[0] += command.value
                    elif command.direction == 'z':
                        new_pose[2] += command.value
                    elif command.direction == 'rx':
                        new_pose[3] += command.value
                    elif command.direction == 'ry':
                        new_pose[4] += command.value
                    elif command.direction == 'rz':
                        new_pose[5] += command.value
                    
                    self.servoL(new_pose, target_pose_as_next_current_pose=True, use_movejp=False)
                
            except Exception as e:
                # print(f"Error: {e}")
                time.sleep(0.01)  # 没有新命令时短暂休眠

def main():
    command_queue = Queue()
    
    # 创建并启动控制进程
    keyboard_controller = KeyboardController(command_queue)
    robot_controller = RobotControlProcess(command_queue)
    
    keyboard_controller.start()
    robot_controller.start()
    
    # 等待进程结束
    keyboard_controller.join()
    robot_controller.terminate()
    
if __name__ == "__main__":
    main()