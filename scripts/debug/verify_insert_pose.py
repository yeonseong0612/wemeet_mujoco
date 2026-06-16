"""수정된 건-tip IK 로 insert 자세를 풀고 단일 렌더로 시각 확인."""
import os, sys, numpy as np, mujoco
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.ik import get_joint_ids, solve_pose_ik

R_TIP_IN_OBJ = np.array([[0,0,1.0],[0,1.0,0],[-1.0,0,0]])
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "models", "scene.xml"))
d = mujoco.MjData(m)
jids = get_joint_ids(m, ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"])
if m.nkey > 0: mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

bid = m.body("charge_port_frame").id
R_port = d.xmat[bid].reshape(3, 3).copy()
tip = m.site("charger_tip_site").id

r = mujoco.Renderer(m, height=480, width=640)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for i, (site, cam, ttl) in enumerate([("charge_port_insert", "overview_camera", "insert (overview)"),
                                       ("charge_port_insert", "cam_port", "insert (cam_port)")]):
    sid = m.site(site).id
    if m.nkey > 0: mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    tpos = d.site_xpos[sid].copy()
    trot = R_port @ R_TIP_IN_OBJ
    solve_pose_ik(m, d, tip, tpos, trot, jids, max_iters=600, step_size=0.3, verbose=False)
    r.update_scene(d, camera=cam)
    ax[i].imshow(r.render()); ax[i].set_title(ttl); ax[i].axis("off")
out = os.path.join(ROOT, "outputs", "insert_pose_check.png")
plt.tight_layout(); plt.savefig(out, dpi=110); print("saved:", out)
