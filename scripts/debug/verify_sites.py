import os, numpy as np, mujoco
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "models", "scene.xml"))
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
np.set_printoptions(precision=4, suppress=True)
for s in ["charge_port_center", "charge_port_approach", "charge_port_insert", "charge_port_axis_site"]:
    print(f"{s:24s} world={d.site_xpos[m.site(s).id]}")
c = d.site_xpos[m.site('charge_port_center').id]
a = d.site_xpos[m.site('charge_port_axis_site').id]
ax = a - c; ax /= np.linalg.norm(ax)
print("insertion axis center->axis_site (world):", ax)

b = m.body('charge_port_frame').id
R = d.xmat[b].reshape(3, 3)
print("\nbody xmat (cols = object X,Y,Z in world):\n", R)
print("object +Z (front normal) in world:", R[:, 2])
print("object -Z (insert dir)   in world:", -R[:, 2])
print("body_quat:", m.body_quat[b])
