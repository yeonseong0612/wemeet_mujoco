"""EE.obj 충전건 메시의 실제 형상을 custom_ee 프레임에서 확인.

geom 변환: pos="0.08 0.1 -0.03" euler="1.5708 0 0", mesh scale 0.001.
charger_tip_site(0.25,0,0) / charger_axis_site(0.35,0,0) 가 실제 총구 끝/축과
일치하는지 검증한다.
"""
import os
import numpy as np
import trimesh

ROOT = "/home/yskim/projects/wemeet_mujoco"
MESH = os.path.join(ROOT, "models/robots/ur10e/assets/EE.obj")


def euler_to_R(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def main():
    mesh = trimesh.load(MESH, force="mesh")
    v = np.asarray(mesh.vertices, dtype=np.float64)

    # 메시 native AABB (스케일 전)
    print("=== EE.obj native (mesh local, pre-scale) ===")
    print("  AABB min:", np.round(v.min(0), 2))
    print("  AABB max:", np.round(v.max(0), 2))
    print("  size    :", np.round(v.max(0) - v.min(0), 2))

    # geom 변환 적용: scale 0.001 → euler(1.5708,0,0) → pos
    v_s = v * 0.001
    R = euler_to_R(1.5708, 0.0, 0.0)
    pos = np.array([0.08, 0.1, -0.03])
    v_ee = (R @ v_s.T).T + pos   # custom_ee 프레임 좌표

    print("\n=== custom_ee 프레임에서의 충전건 메시 ===")
    lo, hi = v_ee.min(0), v_ee.max(0)
    print("  AABB min:", np.round(lo, 4))
    print("  AABB max:", np.round(hi, 4))
    size = hi - lo
    print("  size    :", np.round(size, 4))
    print("  centroid:", np.round(v_ee.mean(0), 4))

    long_axis = int(np.argmax(size))
    print(f"\n  가장 긴 축 = custom_ee {'XYZ'[long_axis]} (length {size[long_axis]:.4f} m)")

    # 긴 축 양 끝의 단면 폭(총구는 가늘어진다) → 어느 끝이 tip인지 추정
    axis_vals = v_ee[:, long_axis]
    for end, sel in [("min단", axis_vals < lo[long_axis] + 0.01),
                     ("max단", axis_vals > hi[long_axis] - 0.01)]:
        pts = v_ee[sel]
        others = [a for a in range(3) if a != long_axis]
        w = pts[:, others].max(0) - pts[:, others].min(0)
        print(f"  {end} (axis={'XYZ'[long_axis]}): 단면폭 {np.round(w,4)}  "
              f"중심 {np.round(pts.mean(0),4)}")

    print("\n=== 현재 site 위치 (custom_ee 프레임) ===")
    print("  charger_tip_site : [0.25, 0, 0]")
    print("  charger_axis_site: [0.35, 0, 0]")
    print("  → 두 site 는 custom_ee +X 축을 총구 축으로 가정함")


if __name__ == "__main__":
    main()
