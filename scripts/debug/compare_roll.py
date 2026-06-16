"""insert 자세: 현재 R_TIP_IN_OBJ vs 롤180° vs 삽입축 반대 비교 렌더."""
import os, sys
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/yskim/projects/wemeet_mujoco"
sys.path.insert(0, ROOT)
from src.ik import get_joint_ids, solve_position_ik, solve_pose_ik

R_BASE = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
ROLL180 = np.diag([1.0, -1.0, -1.0])          # tip+X 축 기준 180° 롤
variants = {
    "current": R_BASE,
    "roll180": R_BASE @ ROLL180,
}
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

model = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "models/scene.xml"))
data = mujoco.MjData(model)
ee = None
renderer = mujoco.Renderer(model, height=480, width=640)

fig, ax = plt.subplots(1, len(variants), figsize=(8 * len(variants), 6))
for r, (name, Rt) in enumerate(variants.items()):
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ee = model.site("charger_tip_site").id
    jids = get_joint_ids(model, JOINTS)
    sid = model.site("charge_port_insert").id
    tgt_pos = data.site_xpos[sid].copy()
    tgt_rot = data.site_xmat[sid].reshape(3, 3).copy() @ Rt
    solve_position_ik(model, data, ee, tgt_pos, jids, max_iters=400, verbose=False)
    solve_pose_ik(model, data, ee, tgt_pos, tgt_rot, jids, max_iters=500,
                  step_size=0.3, damping=1e-4, rot_weight=1.5, verbose=False)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.55, 0.6]
    cam.distance, cam.azimuth, cam.elevation = 1.1, 200, -18
    renderer.update_scene(data, camera=cam)
    ax[r].imshow(renderer.render()); ax[r].axis("off")
    ax[r].set_title(name, fontsize=16)

out = os.path.join(ROOT, "outputs/compare_roll.png")
plt.tight_layout(); plt.savefig(out, dpi=120)
print("saved:", out)
