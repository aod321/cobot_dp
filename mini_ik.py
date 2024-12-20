#%%
from roboticstoolbox.robot import DHRobot
from roboticstoolbox.robot.DHLink import RevoluteMDH,RevoluteDH
import numpy as np

def create_rm65b_robot():
    """
    创建RM65-B机器人模型，使用改进的DH参数(MDH)
    返回: DHRobot对象
    """
    # 将角度转换为弧度
    deg2rad = np.pi / 180
    links = [
        RevoluteMDH(
            a=0.0, 
            alpha=0.0,
            d=0.2405,
            offset=0.0,
            qlim=np.array([-178, 178]) * deg2rad
        ),
        RevoluteMDH(
            a=0.0, 
            alpha=90*deg2rad,
            d=0.0,
            offset=90.0*deg2rad,
            qlim=np.array([-130, 130]) * deg2rad
        ),
        RevoluteMDH(
            a=0.256, 
            alpha=0.0,
            d=0.0,
            offset=90.0*deg2rad,
            qlim=np.array([-135, 135]) * deg2rad
        ),
        RevoluteMDH(
            a=0.0, 
            alpha=90.0*deg2rad,
            d=0.210,
            offset=0,
            qlim=np.array([-178, 178]) * deg2rad
        ),
        RevoluteMDH(
            a=0.0, 
            alpha=-90.0*deg2rad,
            d=0.0,
            offset=0.0,
            qlim=np.array([-128, 128]) * deg2rad
        ),
        RevoluteMDH(
            a=0.0, 
            alpha=90.0*deg2rad,
            d=0.144,
            offset=0.0,
            qlim=np.array([-360, 360]) * deg2rad
        )
    ]

    robot = DHRobot(links, name="RM65-B", manufacturer="RM")
    robot.qz = np.array([0]*robot.n)
    robot.qr = np.array([-15.805, 1.628, -104.147, 6.648, -70.063, 20.478]) * deg2rad
    return robot

# %%
if __name__ == "__main__":
    cobot_model = create_rm65b_robot()
    num_joints = cobot_model.n
    q = np.array([0]*num_joints)
    # %%
    # q=0的状态下进行正解计算
    cobot_model.plot(cobot_model.qr)
    print(cobot_model)

    # %%
    import numpy as np
    deg2rad = np.pi / 180
    # Convert joint angles from degrees to radians
    q_target_gt = np.array([-10.361, 2.327, -104.474, -9.396, -73.217, 20.445]) * deg2rad

    # Forward kinematics to get end-effector pose
    Tep_target = cobot_model.fkine(q_target_gt)
    print("\nTarget end-effector pose:")
    print(Tep_target)

    # Visualize the robot at target configuration
    cobot_model.plot(q_target_gt)

    # %% 逆解
    # 根据Tep_target，求解关节角度solve_q_target
    import roboticstoolbox as rtb
    from spatialmath import SE3
    rad2deg = 180 / np.pi
    sol = cobot_model.ik_LM(Tep_target)         # solve IK
    print(sol[0] * rad2deg)
    print(f"IK Error: {np.linalg.norm(sol[0] - q_target_gt)}")
    qt = rtb.jtraj(cobot_model.qr, sol[0], 50)
    cobot_model.plot(qt.q, backend='pyplot', movie='panda1.gif')
    # %%

