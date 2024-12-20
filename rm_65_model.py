#%%
from roboticstoolbox.robot import DHRobot
from roboticstoolbox.robot.DHLink import RevoluteMDH,RevoluteDH
import numpy as np

class RM65(DHRobot):
    def __init__(self):
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
        super().__init__(links, name="RM65-B", manufacturer="RealMan")
        self.qz = np.array([0]*self.n)
        self.qr = np.array([-15.805, 1.628, -104.147, 6.648, -70.063, 20.478]) * deg2rad
        
# %%
if __name__ == "__main__":
    robot = RM65()
    print(robot)
