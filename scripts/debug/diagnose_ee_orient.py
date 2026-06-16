"""충전건(EE) 삽입 자세 진단: 현재 방식 vs charger_tip_site 정렬 후보 roll 비교."""
import os, sys, numpy as np, mujoco
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.ik import get_joint_ids, solve_pose_ik

m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "models", "scene.xml"))
d = mujoco.MjData(m)
np.set_printoptions(precision=3, suppress=True)
jn = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"]
jids = get_joint_ids(m, jn)

def reset():
    if m.nkey > 0: mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

reset()
# 포트 프레임/축 (world)
bid = m.body("charge_port_frame").id
R_port = d.xmat[bid].reshape(3, 3).copy()
center = d.site_xpos[m.site("charge_port_center").id].copy()
axis_pt = d.site_xpos[m.site("charge_port_axis_site").id].copy()
insert = d.site_xpos[m.site("charge_port_insert").id].copy()
d_in = axis_pt - center; d_in /= np.linalg.norm(d_in)   # 포트 안쪽(삽입) 방향
port_up = R_port[:, 1].copy()                            # object Y = AC 위
print("port d_in(insert dir):", d_in, " port_up:", port_up)

tip_id = m.site("charger_tip_site").id
axs_id = m.site("charger_axis_site").id
ee_id = m.site("attachment_site").id

def gun_axis_world():
    return (d.site_xpos[axs_id] - d.site_xpos[tip_id]) / np.linalg.norm(d.site_xpos[axs_id] - d.site_xpos[tip_id])

def tip_R():
    return d.site_xmat[tip_id].reshape(3, 3).copy()

# --- 현재 방식: attachment_site -> R_port ---
reset()
solve_pose_ik(m, d, ee_id, insert, R_port, jids, max_iters=500, step_size=0.3, verbose=False)
print("\n[현재방식 attachment_site->R_port]")
print("  gun axis . d_in =", float(np.dot(gun_axis_world(), d_in)), "(1이면 삽입정렬)")
print("  tip pos err to insert =", np.linalg.norm(d.site_xpos[tip_id] - insert))

# --- 후보: charger_tip_site 정렬, roll 0/90/180/270 about gun-X(=d_in) ---
def Rx(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
# tip 목표: X=d_in, Y=port_up(그람슈미트), Z=X×Y
X = d_in
Y = port_up - np.dot(port_up, X) * X; Y /= np.linalg.norm(Y)
Z = np.cross(X, Y)
R_base = np.column_stack([X, Y, Z])

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, roll in enumerate([0, 90, 180, 270]):
    R_t = R_base @ Rx(np.deg2rad(roll))
    reset()
    _,_,_,pe,re = solve_pose_ik(m, d, tip_id, insert, R_t, jids, max_iters=600, step_size=0.3, verbose=False)
    ga = float(np.dot(gun_axis_world(), d_in))
    gun_up = tip_R()[:, 1]   # tip +Y
    up_dot = float(np.dot(gun_up, port_up))
    print(f"[tip roll={roll:3d}] gun_axis.d_in={ga:+.3f}  tipY.port_up={up_dot:+.3f}  ik_pos_err={pe*1e3:.1f}mm")
    r = mujoco.Renderer(m, height=480, width=640)
    r.update_scene(d, camera="overview_camera")
    axes[i].imshow(r.render()); axes[i].set_title(f"tip roll {roll}"); axes[i].axis("off")
out = os.path.join(ROOT, "outputs", "ee_orient_candidates.png")
plt.tight_layout(); plt.savefig(out, dpi=100); print("saved:", out)
