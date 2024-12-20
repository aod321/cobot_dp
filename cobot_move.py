# %%
import roboticstoolbox as rtb
from roboticstoolbox import DHRobot, RevoluteMDH
import numpy as np
from rm_65_model import RM65

robot = RM65()
print(robot)
# %% ctraj固定点数插值
from spatialmath import SE3
import numpy as np

deg2rad = np.pi/180
current_joint = np.array([[-5.833, -23.236, -109.084, -0.504, -41.73, 26.942]])*deg2rad
T0 = robot.fkine(current_joint)  # 起始位姿

# 构造目标位姿 (保持姿态不变,只改变位置)
target_pos = T0.t + [0.1, 0, 0]  # 在起始位置基础上，沿X轴移动100mm
T1 = SE3(target_pos) * SE3.RPY(T0.rpy())

# 使用ctraj生成笛卡尔空间的轨迹
traj = rtb.ctraj(T0, T1, 50) # 生成50个点

min_manip = float('inf')  # 记录最小操作度

# 计算逆解,# 计算逆解前先检查轨迹上的操作度
q_traj = []
q_prev = current_joint[0]
for i,T in enumerate(traj):
    q = robot.ik_LM(T, q0=q_prev)
    q_traj.append(q[0])
    q_prev = q[0]
q_traj = np.array(q_traj)

# 绘制轨迹
robot.plot(q_traj, backend='pyplot', movie='rm65_move_ctraj.gif')

# %%  用mstraj, 根据速度和加速度要求生成轨迹
current_pos = T0.t

# 定义路径点 - 起点和终点
viapoints = np.array([
    current_pos,                    # 起点
    current_pos + [0.05, 0, 0],    # 中间点
    current_pos + [0.1, 0, 0]      # 终点
])

# 生成轨迹
traj = rtb.mstraj(
    viapoints,      # 路径点
    dt=0.05,        # 时间步长(s)
    tacc=0.2,       # 加速时间(s)
    qdmax=0.5,      # 最大速度(m/s)
)
# traj.t 包含时间序列
# traj.q 包含位置序列
# 可以通过循环来执行轨迹:
q_traj = []
q_prev = current_joint[0]
for pos in traj.q:
    # 构造目标位姿
    T = SE3(pos) * SE3.RPY(T0.rpy())  # 保持当前旋转姿态不变
    # 求解逆运动学
    q = robot.ik_LM(T,q0=q_prev)
    # 保存轨迹
    q_traj.append(q[0])
    q_prev = q[0]
q_traj = np.array(q_traj)

robot.plot(q_traj, backend='pyplot', movie='rm65_mstraj.gif')

# %%结合ctraj和mstraj,用ctraj生成10个点数,用mstraj生成中间点
from spatialmath import SE3
import numpy as np

deg2rad = np.pi/180
current_joint = np.array([[-5.833, -23.236, -109.084, -0.504, -41.73, 26.942]])*deg2rad
T0 = robot.fkine(current_joint)  # 起始位姿
target_pos = T0.t + [1, 0, 0]  # 在起始位置基础上，沿X轴移动1000mm
T1 = SE3(target_pos) * SE3.RPY(T0.rpy())

# 首先用ctraj生成关键路径点(直接插值)
traj_points = rtb.ctraj(T0, T1, 10)  # 生成10个关键点
via_points = np.array([T.t for T in traj_points])  # 提取位置信息

# 使用mstraj在关键点之间生成平滑轨迹
smooth_traj = rtb.mstraj(
    via_points,     # 路径点
    dt=0.05,        # 时间步长(s)
    tacc=0.2,       # 加速时间(s)
    qdmax=0.5,      # 最大速度(m/s)
)

# 生成关节空间轨迹
q_traj = []
q_prev = current_joint[0]
for pos in smooth_traj.q:
    # 构造目标位姿，保持初始姿态
    T = SE3(pos) * SE3.RPY(T0.rpy())
    # 求解逆运动学
    q = robot.ik_LM(T, q0=q_prev,
                    joint_limits=True)
    success = q[1]
    if not success:
        print("轨迹点超出机械臂工作范围，停止运动")
        break
    q_traj.append(q[0])
    q_prev = q[0]
q_traj = np.array(q_traj)

# 绘制轨迹
robot.plot(q_traj, backend='pyplot', movie='rm65_combined_traj.gif')

# %%
