"""커넥터 총신(아래쪽 돌출부) 점들을 PCA로 피팅해 삽입축과 총구 끝점 산출."""
import os
import numpy as np
import trimesh

ROOT = "/home/yskim/projects/wemeet_mujoco"
MESH = os.path.join(ROOT, "models/robots/ur10e/assets/EE.obj")


def main():
    mesh = trimesh.load(MESH, force="mesh")
    v = np.asarray(mesh.vertices, dtype=np.float64) * 0.001
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    v = (Rx @ v.T).T + np.array([0.08, 0.1, -0.03])

    # 커넥터 총신 = 아래쪽으로 돌출된 좁은 관 (Z 낮고 Y 작은 영역)
    sel = (v[:, 2] < -0.035) & (v[:, 1] < 0.06)
    barrel = v[sel]
    print(f"barrel pts: {len(barrel)}")

    c = barrel.mean(0)
    cov = (barrel - c).T @ (barrel - c)
    w, V = np.linalg.eigh(cov)
    axis = V[:, np.argmax(w)]          # 최대 분산 = 총신 축
    # 총구 끝(바깥쪽)이 -Y,-Z 방향이 되도록 부호 정렬
    if axis[1] > 0 or axis[2] > 0:
        axis = -axis
    axis /= np.linalg.norm(axis)

    proj = (barrel - c) @ axis
    tip = c + axis * proj.max()        # 축 방향 가장 바깥 = 총구 끝
    print("커넥터 총신 축(삽입방향, custom_ee):", np.round(axis, 4))
    print("총구 끝점(custom_ee)             :", np.round(tip, 4))
    print("총신 길이                        :", round(proj.max() - proj.min(), 4))

    # 제안 site 위치
    tip_site = tip
    axis_site = tip + axis * 0.10      # 끝점에서 축 따라 +0.10m (기존 tip→axis 간격 유지)
    print("\n제안:")
    print(f"  charger_tip_site  pos = {np.round(tip_site,4).tolist()}")
    print(f"  charger_axis_site pos = {np.round(axis_site,4).tolist()}")
    print(f"  (axis_site 는 tip 에서 삽입방향 반대로 0.10m → IK 축 정의용)")

    # 반대편(축 안쪽)도 출력 — axis_site 를 안쪽에 둘지 결정용
    axis_site_in = tip - axis * 0.10
    print(f"  대안 axis_site(안쪽) pos = {np.round(axis_site_in,4).tolist()}")


if __name__ == "__main__":
    main()
