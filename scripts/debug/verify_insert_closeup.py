"""insert 자세 단일 클로즈업 — 총구가 충전구에 결합하는지 확인."""
import os, sys
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/yskim/projects/wemeet_mujoco"
sys.path.insert(0, ROOT)
from src.ik import get_joint_ids, solve_position_ik, solve_pose_ik

R_TIP_IN_OBJ = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

model = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "models/scene.xml"))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
ee = model.site("charger_tip_site").id
jids = get_joint_ids(model, JOINTS)

sid = model.site("charge_port_insert").id
tgt_pos = data.site_xpos[sid].copy()
tgt_rot = data.site_xmat[sid].reshape(3, 3).copy() @ R_TIP_IN_OBJ
solve_position_ik(model, data, ee, tgt_pos, jids, max_iters=400, verbose=False)
solve_pose_ik(model, data, ee, tgt_pos, tgt_rot, jids, max_iters=500,
              step_size=0.3, damping=1e-4, rot_weight=1.5, verbose=False)

renderer = mujoco.Renderer(model, height=480, width=640)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for k, (az, el, d) in enumerate([(150, -10, 0.45), (210, -8, 0.45)]):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.8, 0.585]
    cam.distance, cam.azimuth, cam.elevation = d, az, el
    renderer.update_scene(data, camera=cam)
    ax[k].imshow(renderer.render()); ax[k].axis("off")
    ax[k].set_title(f"insert (az={az})")
out = os.path.join(ROOT, "outputs/verify_insert_closeup.png")
plt.tight_layout(); plt.savefig(out, dpi=120)
print("saved:", out)
