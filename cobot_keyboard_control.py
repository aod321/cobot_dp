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

@dataclass
class MoveCommand:
    direction: str  # 'x', 'y', 'z'
    value: float
    
class KeyboardController(mp.Process):
    def __init__(self, command_queue: Queue):
        super().__init__()
        self.command_queue = command_queue
        self.step_size = 0.01  # 每次移动的步长(米)
        
    def run(self):
        print("键盘控制已启动")
        print("使用以下按键控制:")
        print("X轴: A/D")
        print("Y轴: W/S")
        print("Z轴: Q/E")
        print("按ESC退出")
        
        # 创建一个小窗口来接收键盘输入
        cv2.namedWindow('Robot Control', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Robot Control', 300, 100)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('a'):
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
            elif key == 27:  # ESC键
                break
                
        cv2.destroyAllWindows()

class RobotControlProcess(mp.Process):
    def __init__(self, command_queue: Queue):
        super().__init__()
        self.command_queue = command_queue
        self.cobot_controller = RobotController()
        self.rm_65_ik_model = RM65()
        self.cobot_controller.connect()
        self.cobot_controller.speed_upbound = 50
        # 获取初始位姿
        self.current_pose = self.cobot_controller.calc_end_pose()
        
    def moveL(self, target_pose, target_pose_as_next_current_pose=False):
        self.cobot_controller.moveL(target_pose, speed=30)
        if target_pose_as_next_current_pose:
            self.current_pose = target_pose
        else:
            self.current_pose = self.cobot_controller.calc_end_pose()
        
    def run(self):
        print("机械臂控制已启动")
        
        while True:
            try:
                command = self.command_queue.get_nowait()
                new_pose = self.current_pose.copy()
                
                if command.direction == 'x':
                    new_pose[0] += command.value
                elif command.direction == 'y':
                    new_pose[1] += command.value
                elif command.direction == 'z':
                    new_pose[2] += command.value
                    
                self.moveL(new_pose)
                
            except Exception:
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