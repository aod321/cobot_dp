# %%
import roboticstoolbox as rtb
panda_dh = rtb.models.DH.Panda()
print(panda_dh)
panda_dh.plot(panda_dh.qr)
Tep = SE3.Trans(0.6, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
sol = panda_dh.ik_LM(Tep)
print(sol)
q_pickup = sol[0]
print(panda_dh.fkine(q_pickup))

qt = rtb.jtraj(panda_dh.qr, q_pickup, 50)
panda_dh.plot(qt.q, backend='pyplot', movie='panda1.gif')
# %%
# Test IK solutions along x-axis
import numpy as np
from spatialmath import SE3

# Create test range from -0.5 to 0.5 with step 0.1
x_range = np.arange(-0.5, 2, 0.5)

for x in x_range:
    # Create target pose with varying x position
    Tep = SE3.Trans(0.6 + x, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
    
    # Solve IK
    sol = panda_dh.ik_LM(Tep)
    if sol[0] is not None:
        # Generate trajectory
        qt = rtb.jtraj(panda_dh.qr, sol[0], 50)
        # Plot and save animation
        panda_dh.plot(qt.q, backend='pyplot', movie=f'panda_x_{x:.1f}.gif')
    else:
        print(f"Failed to find IK solution for x = {x:.1f}")


# %%
