"""정상화된 scene에서 IK 수렴/축 정렬 headless 검증 (viewer 없음)."""
import os, sys
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.ik import get_joint_ids, solve_pose_ik

np.set_printoptions(precision=4, suppress=True)

m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "models", "scene.xml"))
d = mujoco.MjData(m)
if m.nkey > 0:
    mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
joint_ids = get_joint_ids(m, joint_names)
ee = m.site("attachment_site").id

# 충전구 삽입축(world): center -> axis_site
c = d.site_xpos[m.site("charge_port_center").id].copy()
ax = d.site_xpos[m.site("charge_port_axis_site").id].copy() - c
ax /= np.linalg.norm(ax)
print("port insertion axis (world):", ax)
print("EE start pos:", d.site_xpos[ee].copy())

for name in ["charge_port_approach", "charge_port_insert"]:
    sid = m.site(name).id
    tpos = d.site_xpos[sid].copy()
    trot = d.site_xmat[sid].reshape(3, 3).copy()
    q, fpos, frot, pe, re = solve_pose_ik(
        m, d, ee, tpos, trot, joint_ids,
        max_iters=500, pos_tol=1e-4, rot_tol=1e-3,
        step_size=0.3, damping=1e-4, rot_weight=0.5, verbose=True)
    eez = frot[:, 2]
    print(f"[{name}] pos_err={pe:.5f}  rot_err={re:.5f}  "
          f"reached={np.linalg.norm(fpos - tpos) < 2e-3}")
    print(f"   EE +Z . port_axis = {np.dot(eez, ax):+.3f}   "
          f"EE -Z . port_axis = {np.dot(-eez, ax):+.3f}")
    mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
