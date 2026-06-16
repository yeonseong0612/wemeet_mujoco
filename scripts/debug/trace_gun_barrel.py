"""충전건 커넥터(총구) 끝과 삽입축을 custom_ee 프레임에서 정밀 추적."""
import os
import numpy as np
import trimesh

ROOT = "/home/yskim/projects/wemeet_mujoco"
MESH = os.path.join(ROOT, "models/robots/ur10e/assets/EE.obj")


def euler_to_R(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rx  # ry=rz=0


def main():
    mesh = trimesh.load(MESH, force="mesh")
    v = np.asarray(mesh.vertices, dtype=np.float64) * 0.001
    R = euler_to_R(1.5708, 0, 0)
    v = (R @ v.T).T + np.array([0.08, 0.1, -0.03])

    lo, hi = v.min(0), v.max(0)
    print("AABB custom_ee:  min", np.round(lo, 4), " max", np.round(hi, 4))

    # Y축(긴 축)을 따라 슬라이스하며 단면 중심/반경 추적
    print("\nY-slice  | center(X,Z)         | radius  | npts")
    ys = np.linspace(lo[1], hi[1], 24)
    prev = None
    centers = []
    for y0, y1 in zip(ys[:-1], ys[1:]):
        sel = (v[:, 1] >= y0) & (v[:, 1] < y1)
        if sel.sum() < 5:
            continue
        pts = v[sel]
        c = pts[:, [0, 2]].mean(0)
        r = np.linalg.norm(pts[:, [0, 2]] - c, axis=1).mean()
        centers.append((0.5 * (y0 + y1), c[0], c[1], r))
        print(f" {0.5*(y0+y1):+.3f}  | ({c[0]:+.3f}, {c[1]:+.3f})     | {r:.4f} | {sel.sum()}")

    centers = np.array(centers)
    # 가장 가는(반경 최소) 슬라이스 = 커넥터 총구 추정
    imin = int(np.argmin(centers[:, 3]))
    print(f"\n반경 최소 슬라이스: Y={centers[imin,0]:+.3f}  "
          f"center(X,Z)=({centers[imin,1]:+.3f},{centers[imin,2]:+.3f})  r={centers[imin,3]:.4f}")

    # 끝단(Y 최소/최대) 비교
    print(f"\nY-min 끝단 center: X,Z=({centers[0,1]:+.3f},{centers[0,2]:+.3f}) r={centers[0,3]:.4f}")
    print(f"Y-max 끝단 center: X,Z=({centers[-1,1]:+.3f},{centers[-1,2]:+.3f}) r={centers[-1,3]:.4f}")


if __name__ == "__main__":
    main()
