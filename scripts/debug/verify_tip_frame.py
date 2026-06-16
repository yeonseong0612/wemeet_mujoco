"""제안하는 charger_tip_site 프레임 + R_TIP_IN_OBJ 가 총신을 삽입방향에 정렬하는지 수치 검증."""
import numpy as np

# 총신(barrel) forward in custom_ee (PCA 결과)
ex = np.array([0.0, -0.8746, -0.4849]); ex /= np.linalg.norm(ex)
ey = np.array([1.0, 0.0, 0.0])
ez = np.cross(ex, ey); ez /= np.linalg.norm(ez)
ey = np.cross(ez, ex)                       # 직교화
R_site_in_ee = np.column_stack([ex, ey, ez])
print("R_site_in_ee (cols ex,ey,ez):\n", np.round(R_site_in_ee, 3))
print("det:", round(np.linalg.det(R_site_in_ee), 4))

# 제안 R_TIP_IN_OBJ
R_TIP_IN_OBJ = np.array([[0.0, -1.0, 0.0],
                          [0.0,  0.0, 1.0],
                          [-1.0, 0.0, 0.0]])
print("\nR_TIP_IN_OBJ det:", round(np.linalg.det(R_TIP_IN_OBJ), 4))

# 포트(approach) site world rotation (앞서 덤프한 값) cols +X,+Y,+Z
R_port_world = np.array([[1.0, 0.0,  0.0],
                          [0.0, 0.0, -1.0],
                          [0.0, 1.0,  0.0]])

# IK 수렴 시 charger_tip_site world frame = target
R_target = R_port_world @ R_TIP_IN_OBJ
print("\nR_target (tip site world at insert):\n", np.round(R_target, 3))

# 총신 forward = site local +X → world
barrel_world = R_target @ np.array([1.0, 0.0, 0.0])
print("\n총신 forward (world):", np.round(barrel_world, 3), "  (기대: [0,1,0]=삽입 +Y)")

tip_up_world = R_target @ np.array([0.0, 0.0, 1.0])
print("총구 up (world)    :", np.round(tip_up_world, 3), "  (기대: [0,0,1]=위)")

insertion_dir = np.array([0.0, 1.0, 0.0])
print("\nbarrel·insertion =", round(float(barrel_world @ insertion_dir), 4), " (기대 +1)")

# 제안 site 위치
tip = np.array([0.0005, -0.0256, -0.0823])
axis_site = tip + ex * 0.10
print("\ncharger_tip_site  pos =", np.round(tip, 4).tolist())
print("charger_axis_site pos =", np.round(axis_site, 4).tolist())
print("xyaxes =", " ".join(f"{x:.4f}" for x in np.r_[ex, ey]))
