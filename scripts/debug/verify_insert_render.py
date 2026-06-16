"""수정된 charger_tip_site + R_TIP_IN_OBJ 로 IK 를 풀어 삽입 자세를 렌더링 검증."""
import os
import sys
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/yskim/projects/wemeet_mujoco"
sys.path.insert(0, ROOT)
from src.ik import get_joint_ids, solve_position_ik, solve_pose_ik

R_TIP_IN_OBJ = np.array([[0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0]])
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

model = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "models/scene.xml"))
data = mujoco.MjData(model)
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

ee = model.site("charger_tip_site").id
jids = get_joint_ids(model, JOINTS)
qpos_ids = [model.jnt_qposadr[j] for j in jids]

renderer = mujoco.Renderer(model, height=480, width=640)

cam = mujoco.MjvCamera()
cam.lookat[:] = [0.0, 0.79, 0.585]
cam.distance = 0.6
cam.azimuth = 135
cam.elevation = -12

frames = {}
renderer.update_scene(data, camera=cam)
frames["start"] = renderer.render().copy()

for site in ["charge_port_approach", "charge_port_insert"]:
    sid = model.site(site).id
    tgt_pos = data.site_xpos[sid].copy()
    tgt_rot = data.site_xmat[sid].reshape(3, 3).copy() @ R_TIP_IN_OBJ

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    solve_position_ik(model, data, ee, tgt_pos, jids, max_iters=400, verbose=False)
    _, fpos, frot, pe, re = solve_pose_ik(
        model, data, ee, tgt_pos, tgt_rot, jids, max_iters=500,
        step_size=0.3, damping=1e-4, rot_weight=1.5, verbose=False)

    # 총신 forward (tip site +X) vs 삽입축
    barrel = frot[:, 0]
    port_axis = tgt_rot[:, 0]   # tip+X 목표 = 삽입방향
    print(f"[{site}] pos_resid={pe*1e3:.2f} mm  rot_resid={re:.4f}  "
          f"tip_pos_err={np.linalg.norm(fpos-tgt_pos)*1e3:.2f} mm  "
          f"barrel·target={barrel@port_axis:.4f}")

    renderer.update_scene(data, camera=cam)
    frames[site] = renderer.render().copy()

fig, ax = plt.subplots(1, 3, figsize=(21, 6))
for k, (name, img) in enumerate(frames.items()):
    ax[k].imshow(img); ax[k].set_title(name); ax[k].axis("off")
out = os.path.join(ROOT, "outputs/verify_insert.png")
plt.tight_layout(); plt.savefig(out, dpi=110)
print("saved:", out)
