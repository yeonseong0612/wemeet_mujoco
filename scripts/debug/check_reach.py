import os, sys, numpy as np, mujoco
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.ik import get_joint_ids, solve_position_ik, solve_pose_ik
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "models", "scene.xml"))
d = mujoco.MjData(m)
np.set_printoptions(precision=4, suppress=True)

jn = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"]
jids = get_joint_ids(m, jn)
ee = m.site("attachment_site").id

def reset():
    if m.nkey > 0: mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

reset()
approach = d.site_xpos[m.site("charge_port_approach").id].copy()

# 1) position-only IK to actual port-approach (base 바로 위)
reset()
q, fp, pe = solve_position_ik(m, d, ee, approach, jids, max_iters=800, verbose=False)
print(f"[pos-only -> port approach {approach}]  pos_err={pe:.5f}  reached={pe<1e-3}")

# 2) position-only IK to a target offset IN FRONT of the base (현실적 충전 배치 가정)
front = np.array([0.6, 0.0, 0.4])
reset()
q, fp, pe = solve_position_ik(m, d, ee, front, jids, max_iters=800, verbose=False)
print(f"[pos-only -> front target  {front}]  pos_err={pe:.5f}  reached={pe<1e-3}")

# 3) pose IK to front target with EE -Z pointing +X (삽입축 모사)
trot = np.array([[0,0,1.0],[0,1.0,0],[-1.0,0,0]])  # EE z -> +X
reset()
q, fp, fr, pe, re = solve_pose_ik(m, d, ee, front, trot, jids, max_iters=800,
                                  pos_tol=1e-4, rot_tol=1e-3, step_size=0.3,
                                  damping=1e-4, rot_weight=0.5, verbose=False)
print(f"[pose -> front target +Xaxis]  pos_err={pe:.5f}  rot_err={re:.5f}  reached={pe<2e-3}")
