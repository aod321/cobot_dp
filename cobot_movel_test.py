#%%
from rm_65_model import RM65
from cobot_controller import RobotController
rm_65_ik_model = RM65()
cobot_controller = RobotController()

# %%
cobot_controller.connect()
# %%
import numpy as np
from spatialmath import SE3
import roboticstoolbox as rtb
deg2rad = np.pi/180
rad2deg = 180 / np.pi
current_joint_angles = cobot_controller.get_joint_angles()
current_joints_rad = np.array(current_joint_angles) * deg2rad
T0 = rm_65_ik_model.fkine(current_joints_rad)  # 起始位姿
current_pos = T0.t
target_pos = T0.t + [0, 0.05, 0.00]  # 在起始位置基础上，沿X轴移动100mm
T1 = SE3(target_pos) * SE3.RPY(T0.rpy())
via_points = np.array([T0.t, T1.t])
# 使用mstraj在关键点之间生成平滑轨迹
smooth_traj = rtb.mstraj(
    via_points,     # 路径点
    dt=0.02,        # 时间步长(s)
    tacc=0.05,       # 加速时间(s)
    qdmax=1,      # 最大速度(m/s)
)
cobot_controller.speed_upbound = 50
# 生成关节空间轨迹
q_traj = []
q_prev = current_joints_rad
for i, pos in enumerate(smooth_traj.q):
    # 构造目标位姿，保持初始姿态
    T = SE3(pos) * SE3.RPY(T0.rpy())
    to_pose = T.t.tolist() + T.rpy().tolist()
    cobot_controller.moveL(to_pose, speed=30, in_trajectory=i != len(smooth_traj.q) - 1)