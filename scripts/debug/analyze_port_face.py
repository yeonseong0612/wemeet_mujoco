"""Task A: Port.obj mating face(소켓 입구) 위치 추정 (조사 전용)."""
import os
import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MESH = os.path.join(ROOT, "models", "assets", "Port.obj")

m = trimesh.load(MESH, force="mesh")
v = m.vertices
np.set_printoptions(precision=2, suppress=True)

zmin, zmax = v[:, 2].min(), v[:, 2].max()
print(f"Z range: [{zmin:.1f}, {zmax:.1f}] mm   (origin z=0)")

# Z 방향 단면 분포: 각 슬랩의 정점 수 + XY 평균
print("\nZ-slab 분포 (앞=+Z → 뒤=-Z):")
edges = np.linspace(zmin, zmax, 13)
for i in range(len(edges) - 1):
    lo, hi = edges[i], edges[i + 1]
    sel = (v[:, 2] >= lo) & (v[:, 2] < hi)
    if sel.sum() == 0:
        continue
    xy = v[sel][:, :2]
    print(f"  z[{lo:7.1f},{hi:7.1f}]  n={sel.sum():5d}  "
          f"x[{xy[:,0].min():6.1f},{xy[:,0].max():6.1f}]  "
          f"y[{xy[:,1].min():6.1f},{xy[:,1].max():6.1f}]  "
          f"xy_mean=({xy[:,0].mean():6.1f},{xy[:,1].mean():6.1f})")

# 앞면(front, +Z 최대 근처 5mm) 정점들의 중심 — mating face 평면 후보
front = v[v[:, 2] > zmax - 5.0]
print(f"\n[front slab] z>{zmax-5:.1f}: n={len(front)}  "
      f"center=({front[:,0].mean():.1f}, {front[:,1].mean():.1f}, {front[:,2].mean():.1f}) mm")
print(f"  front X span [{front[:,0].min():.1f},{front[:,0].max():.1f}]  "
      f"Y span [{front[:,1].min():.1f},{front[:,1].max():.1f}]")

# CCS1: 상단 AC 원형 소켓 + 하단 DC 핀. Y 상단부 형상 확인
print(f"\n전체 AABB center XY = ({(v[:,0].min()+v[:,0].max())/2:.1f}, "
      f"{(v[:,1].min()+v[:,1].max())/2:.1f})")
print(f"origin XY = (0, 0)")
