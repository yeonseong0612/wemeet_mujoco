"""Port.obj 소켓 결합면(2구멍 커넥터)의 body-local 중심/Z 위치 정밀 산출.
소켓은 -Z 를 향함. 커넥터축(X≈0,Y≈-0.015) 근방에서 -Z 방향 입구면을 찾는다."""
import os
import numpy as np
import trimesh

ROOT = "/home/yskim/projects/wemeet_mujoco"
mesh = trimesh.load(os.path.join(ROOT, "models/assets/Port.obj"), force="mesh")
v = np.asarray(mesh.vertices) * 0.001   # body-local (geom euler=0)

print("AABB:", np.round(v.min(0), 4), np.round(v.max(0), 4))

# 커넥터(소켓) 영역: 중앙 컬럼 근방 (현재 site Y=-0.015 근처)
col = (np.abs(v[:, 0]) < 0.03) & (np.abs(v[:, 1] + 0.015) < 0.04)
vc = v[col]
print("\n중앙 컬럼 점수:", len(vc), " Z범위:", round(vc[:,2].min(),4), round(vc[:,2].max(),4))

# Z 슬라이스별 단면적(반경) — 소켓 입구는 단면이 갑자기 커지는 rim
print("\nZ-slice | npts | X-extent | Y-extent | meanXY")
for z0 in np.arange(-0.13, 0.02, 0.01):
    sel = (vc[:, 2] >= z0) & (vc[:, 2] < z0 + 0.01)
    if sel.sum() < 5:
        continue
    p = vc[sel]
    print(f" {z0:+.3f} | {sel.sum():6d} | "
          f"{p[:,0].max()-p[:,0].min():.3f} | {p[:,1].max()-p[:,1].min():.3f} | "
          f"({p[:,0].mean():+.3f},{p[:,1].mean():+.3f})")

# -Z 향한 면(법선) 추정: 면 법선이 -Z 인 삼각형들의 무게중심
if hasattr(mesh, "face_normals"):
    fn = np.asarray(mesh.face_normals)
    fc = np.asarray(mesh.triangles_center) * 0.001
    downz = (fn[:, 2] < -0.7)
    near = downz & (np.abs(fc[:, 0]) < 0.04) & (np.abs(fc[:, 1] + 0.015) < 0.05)
    if near.sum() > 0:
        fcn = fc[near]
        print(f"\n-Z 향한 면 {near.sum()}개  무게중심: {np.round(fcn.mean(0),4)}")
        print(f"  Z 분포: min {fcn[:,2].min():.4f}  max {fcn[:,2].max():.4f}  "
              f"median {np.median(fcn[:,2]):.4f}")
