from rm_65_model import RM65
from cobot_controller import RobotController
import numpy as np
from spatialmath import SE3
import cv2
import time
import roboticstoolbox as rtb

# 初始化机器人和控制器
rm_65_ik_model = RM65()
cobot_controller = RobotController()
cobot_controller.connect()

# 常量定义
deg2rad = np.pi/180
rad2deg = 180/np.pi
STEP_SIZE = 0.05  # 每次移动的距离(米)
MOVE_SPEED = 30   # 运动速度

def get_current_pose():
    """获取当前机器人位姿"""
    current_joint_angles = cobot_controller.get_joint_angles()
    current_joints_rad = np.array(current_joint_angles) * deg2rad
    T0 = rm_65_ik_model.fkine(current_joints_rad)
    return T0
def move_robot(direction):
    """根据方向移动机器人，使用插值生成平滑轨迹"""
    T0 = get_current_pose()
    offset = np.zeros(3)
    
    # 根据按键设置移动方向
    if direction == 'w':      # 前进
        offset[1] = STEP_SIZE
    elif direction == 's':    # 后退
        offset[1] = -STEP_SIZE
    elif direction == 'a':    # 左移
        offset[0] = -STEP_SIZE
    elif direction == 'd':    # 右移
        offset[0] = STEP_SIZE
    elif direction == 'q':    # 上升
        offset[2] = STEP_SIZE
    elif direction == 'e':    # 下降
        offset[2] = -STEP_SIZE
    # 计算目标位置
    target_pos = T0.t + offset
    T1 = SE3(target_pos) * SE3.RPY(T0.rpy())
    
    # 生成平滑轨迹
    via_points = np.array([T0.t, T1.t])
    smooth_traj = rtb.mstraj(
        via_points,     # 路径点
        dt=0.02,        # 时间步长(s)
        tacc=0.05,      # 加速时间(s)
        qdmax=1,        # 最大速度(m/s)
    )

    # 执行平滑轨迹
    for i, pos in enumerate(smooth_traj.q):
        # 构造目标位姿，保持初始姿态
        T = SE3(pos) * SE3.RPY(T0.rpy())
        to_pose = T.t.tolist() + T.rpy().tolist()
        # 最后一个点不需要轨迹连接
        cobot_controller.moveL(to_pose, speed=MOVE_SPEED, in_trajectory=i != len(smooth_traj.q) - 1)

def main():
    print("键盘控制说明：")
    print("W/S: 前进/后退")
    print("A/D: 左移/右移")
    print("Q/E: 上升/下降")
    print("ESC: 退出程序")
    
    cobot_controller.speed_upbound = 50
    
    # 创建一个命名窗口用于接收键盘输入
    cv2.namedWindow('Robot Control', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Robot Control', 300, 100)
    
    try:
        while True:
            # 显示一个简单的图像窗口
            img = np.zeros((100, 300, 3), dtype=np.uint8)
            cv2.putText(img, "Press ESC to exit", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Robot Control', img)
            
            # 等待键盘输入
            key = cv2.waitKey(100) & 0xFF
            
            if key == 27:  # ESC键
                break
            elif key == ord('w'):
                move_robot('w')
            elif key == ord('s'):
                move_robot('s')
            elif key == ord('a'):
                move_robot('a')
            elif key == ord('d'):
                move_robot('d')
            elif key == ord('q'):
                move_robot('q')
            elif key == ord('e'):
                move_robot('e')
                
    except KeyboardInterrupt:
        print("\n程序已终止")
    finally:
        cv2.destroyAllWindows()
        cobot_controller.disconnect()

if __name__ == "__main__":
    main() 