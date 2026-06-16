import os, sys, numpy as np
os.chdir("/home/yskim/projects/wemeet_mujoco")
sys.path.insert(0, ".")
import mujoco

model = mujoco.MjModel.from_xml_path("models/scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

for name in ["charger_tip_site", "charger_axis_site", "attachment_site",
             "charge_port_approach", "charge_port_insert"]:
    sid = model.site(name).id
    pos = data.site_xpos[sid]
    rot = data.site_xmat[sid].reshape(3, 3)
    print(f"{name}")
    print(f"  pos: {np.round(pos, 4)}")
    print(f"  +X: {np.round(rot[:,0], 3)}  +Y: {np.round(rot[:,1], 3)}  +Z: {np.round(rot[:,2], 3)}")

tip = data.site_xpos[model.site("charger_tip_site").id]
ax  = data.site_xpos[model.site("charger_axis_site").id]
d = ax - tip; d /= np.linalg.norm(d)
print(f"\ncharger insertion dir (tip->axis, world): {np.round(d, 3)}")
