"""포트 소켓 입구 방향 vs 총신 forward 를 world 좌표로 비교해 반대결합 원인 진단."""
import os, sys
import numpy as np
import mujoco

ROOT = "/home/yskim/projects/wemeet_mujoco"
sys.path.insert(0, ROOT)
from src.ik import get_joint_ids, solve_position_ik, solve_pose_ik

R_TIP_IN_OBJ = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

model = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "models/scene.xml"))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# 포트 site world 위치
c = data.site_xpos[model.site("charge_port_center").id]
ap = data.site_xpos[model.site("charge_port_approach").id]
ins = data.site_xpos[model.site("charge_port_insert").id]
print("port_center  :", np.round(c, 4))
print("approach     :", np.round(ap, 4), " (총이 시작/접근하는 위치)")
print("insert       :", np.round(ins, 4))
print("approach→insert dir (삽입 진행방향):", np.round((ins - ap) / np.linalg.norm(ins - ap), 3))
print("center→approach dir (소켓 입구가 열린 쪽):", np.round((ap - c) / np.linalg.norm(ap - c), 3))

# Port.obj 소켓 입구 방향 확인: 메시를 body 프레임으로 변환, 핀(돌출부) 방향
import trimesh
mesh = trimesh.load(os.path.join(ROOT, "models/assets/Port.obj"), force="mesh")
v = np.asarray(mesh.vertices) * 0.001   # mm→m, body local (geom pos/euler=0)
print("\nPort.obj body-local AABB: min", np.round(v.min(0), 4), " max", np.round(v.max(0), 4))
print("centroid:", np.round(v.mean(0), 4))
# body +Z 가 소켓 입구쪽이라면(CLAUDE.md) +Z 극단이 입구면
print("body +Z 극단(입구 추정) 평균 Z:", round(v[:, 2].max(), 4),
      " / -Z 극단:", round(v[:, 2].min(), 4))

# 현재 규약으로 IK → 총신 forward(world)
ee = model.site("charger_tip_site").id
jids = get_joint_ids(model, JOINTS)
tgt_pos = ins.copy()
tgt_rot = data.site_xmat[model.site("charge_port_insert").id].reshape(3, 3) @ R_TIP_IN_OBJ
solve_position_ik(model, data, ee, tgt_pos, jids, max_iters=400, verbose=False)
_, fpos, frot, pe, re = solve_pose_ik(model, data, ee, tgt_pos, tgt_rot, jids,
                                      max_iters=500, step_size=0.3, damping=1e-4,
                                      rot_weight=1.5, verbose=False)
print("\n현재 규약 총신 forward(tip +X, world):", np.round(frot[:, 0], 3))
print("삽입 진행방향과 dot:", round(float(frot[:, 0] @ (ins - ap) / np.linalg.norm(ins - ap)), 3))
print("→ +1 이면 총신이 삽입 진행방향과 같음(정상), -1 이면 반대")
