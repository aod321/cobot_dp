from rm_65_model import RM65
from cobot_controller import RobotController
import numpy as np
from spatialmath import SE3
import cv2
import roboticstoolbox as rtb
from interpolation_controller import InterpolationController
from multiprocessing.managers import SharedMemoryManager
from precise_sleep import precise_sleep,precise_wait
import time

def main():
    print("键盘控制说明：")
    print("W/S: 前进/后退")
    print("A/D: 左移/右移")
    print("Q/E: 上升/下降")
    print("ESC: 退出程序")
    deg2rad = np.pi/180
    # 初始化 SharedMemoryManager 和 InterpolationController
    with SharedMemoryManager() as shm_manager:
        with InterpolationController(
            shm_manager=shm_manager,
            robot_ip="192.168.40.102",
            frequency=100,
            verbose=False
        ) as controller:
            # 创建窗口用于接收键盘输入
            cv2.namedWindow('Robot Control', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Robot Control', 300, 100)
            current_pose = controller.get_current_pose()
            t_start = time.monotonic()
            iter_idx = 0
            frequency = 100
            dt = 1/frequency
            command_latency = 0.01
            try:
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # 等待键盘输入
                    key = cv2.waitKey(100) & 0xFF
                    
                    if key == 27:  # ESC键
                        break
                    elif key in [ord(k) for k in 'wsadqe']:
                        # 定义移动步长(米)和旋转步长(弧度)
                        step = 0.01
                        current_pose = controller.get_current_pose()
                        target_pose = current_pose.copy()
                        
                        # 根据按键更新目标位姿
                        if key == ord('w'):  # 前进
                            target_pose[1] += step
                        elif key == ord('s'):  # 后退
                            target_pose[1] -= step
                        elif key == ord('a'):  # 左移
                            target_pose[0] -= step
                        elif key == ord('d'):  # 右移
                            target_pose[0] += step
                        elif key == ord('q'):  # 上升
                            target_pose[2] += step
                        elif key == ord('e'):  # 下降
                            target_pose[2] -= step
                            
                        precise_wait(t_sample)
                        controller.schedule_waypoint(
                            pose=target_pose,
                            target_time=t_command_target-time.monotonic()+time.time()
                        )
                        
            except KeyboardInterrupt:
                print("\n程序已终止")
            finally:
                cv2.destroyAllWindows()

if __name__ == "__main__":
    main()