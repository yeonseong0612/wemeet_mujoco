"""Task A: Port.obj 기하 분석 (조사 전용, 씬/메시 수정 없음)."""
import os
import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MESH = os.path.join(ROOT, "models", "assets", "Port.obj")
SCALE = 0.001  # scene.xml mesh scale (mm -> m)

m = trimesh.load(MESH, force="mesh")
print(f"[load] {MESH}")
print(f"  vertices={len(m.vertices)}  faces={len(m.faces)}  watertight={m.is_watertight}")

v = m.vertices  # in mm (raw obj units)
aabb_min = v.min(axis=0)
aabb_max = v.max(axis=0)
extent = aabb_max - aabb_min
centroid = m.centroid          # area/volume weighted center (mm)
verts_mean = v.mean(axis=0)    # raw vertex mean (mm)
com = m.center_mass if m.is_watertight else None

np.set_printoptions(precision=2, suppress=True)
print("\n=== RAW (mm) ===")
print(f"  mesh origin (obj 0,0,0)   : [0. 0. 0.]   (정의상)")
print(f"  AABB min                  : {aabb_min}")
print(f"  AABB max                  : {aabb_max}")
print(f"  AABB extent (size)        : {extent}")
print(f"  AABB center               : {(aabb_min+aabb_max)/2}")
print(f"  centroid (trimesh)        : {centroid}")
print(f"  vertex mean               : {verts_mean}")
print(f"  center_mass (if watertght): {com}")

# origin이 AABB 안에 있는지: 각 축에서 0이 [min,max] 사이인가
print("\n  origin∈AABB? per-axis:",
      [(float(aabb_min[i]) <= 0.0 <= float(aabb_max[i])) for i in range(3)])
print("  origin relative to AABB min (mm):", -aabb_min)
print("  origin relative to AABB center(mm):", -(aabb_min+aabb_max)/2)

print("\n=== SCALED to meters (x0.001) ===")
print(f"  AABB extent (m)           : {extent*SCALE}")
print(f"  AABB center (m)           : {(aabb_min+aabb_max)/2*SCALE}")
print(f"  centroid (m)              : {centroid*SCALE}")

# 축별 두께로 '얇은 축' = 포트 법선(삽입축) 후보 찾기
thin_axis = int(np.argmin(extent))
print(f"\n  thinnest axis (법선 후보) : {'XYZ'[thin_axis]}  (extent {extent[thin_axis]*SCALE:.4f} m)")
print(f"  longest axes              : {[ 'XYZ'[i] for i in np.argsort(extent)[::-1] ]}")

# 각 면(±각축)에서의 좌표 — mating face 후보 위치
print("\n  face centers along each axis (mm, raw):")
for i in range(3):
    print(f"    {'XYZ'[i]}- face @ {aabb_min[i]:.1f} ,  {'XYZ'[i]}+ face @ {aabb_max[i]:.1f}")
