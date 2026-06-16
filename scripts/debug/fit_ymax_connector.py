"""Ymax 쪽 커넥터(실제 결합부) 끝점과 면 법선(삽입축)을 custom_ee 프레임에서 산출."""
import os
import numpy as np
import trimesh

ROOT = "/home/yskim/projects/wemeet_mujoco"
MESH = os.path.join(ROOT, "models/robots/ur10e/assets/EE.obj")

mesh = trimesh.load(MESH, force="mesh")
v = np.asarray(mesh.vertices, dtype=np.float64) * 0.001
Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
v = (Rx @ v.T).T + np.array([0.08, 0.1, -0.03])

# 커넥터 면 법선: -Y 향한 면이 아니라 +Y(바깥) 향한 캡 삼각형들의 평균 법선
fn = np.asarray(mesh.face_normals).copy()
fn = (Rx @ fn.T).T
fc = (Rx @ (np.asarray(mesh.triangles_center) * 0.001).T).T + np.array([0.08, 0.1, -0.03])

# Ymax 캡 근처(+Y쪽 끝) & 바깥(+Y 성분 큰) 면
near_cap = fc[:, 1] > 0.19
outward = fn[:, 1] > 0.5
capf = near_cap & outward
print(f"Ymax 캡 외향면 수: {capf.sum()}")
normal = fn[capf].mean(0); normal /= np.linalg.norm(normal)
center = fc[capf].mean(0)
print("커넥터 면 법선(삽입축, custom_ee):", np.round(normal, 4))
print("커넥터 면 중심(custom_ee):", np.round(center, 4))

# 면 위 가장 바깥점(법선방향 최대) = 끝점
proj = (v - center) @ normal
tip = center + normal * proj.max()
# 끝점은 면 중심으로 두는 게 결합엔 더 적합 → 면 중심 사용
tip = center.copy()
print("커넥터 끝점(=면중심, custom_ee):", np.round(tip, 4))

axis = normal
ex = axis
ey = np.cross([0, 0, 1.0], ex); ey /= np.linalg.norm(ey)
ez = np.cross(ex, ey)
axis_site = tip + ex * 0.10
print("\n제안:")
print(f"  charger_tip_site  pos   = {np.round(tip, 4).tolist()}")
print(f"  charger_tip_site  xyaxes= {' '.join(f'{x:.4f}' for x in np.r_[ex, ey])}")
print(f"  charger_axis_site pos   = {np.round(axis_site, 4).tolist()}")
print("  ez(up):", np.round(ez, 4))
