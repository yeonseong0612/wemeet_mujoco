#!/usr/bin/env python3
"""
MuJoCo GT depth → FoundationPose 6DoF 자세 추정 검증 스크립트 (Phase 1).

파이프라인:
    models/scene.xml (UR10e + 충전구 Port.obj)
        → cam_port 카메라로 RGB + GT depth(meter) + segmentation mask 추출
        → FoundationPose.register() 로 6DoF pose(ob_in_cam) 추정
        → pose 출력 + 시각화 저장

주의:
    FoundationPose 는 CUDA(nvdiffrast) 가 필수입니다. 이 스크립트는 macOS 등
    CUDA 가 없는 환경에서는 렌더링/마스크 결과만 저장하고 자세 추정은 건너뜁니다.
    실제 6DoF 추정은 GPU 머신에서 실행하세요.

Usage:
    conda activate wemeet
    python scripts/test_foundation_pose.py
"""
import os
import sys

import numpy as np
import mujoco

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH     = os.path.join(PROJECT_ROOT, "models", "scene.xml")
MESH_PATH    = os.path.join(PROJECT_ROOT, "models", "assets", "hummer_charge_port.obj")
FP_DIR       = os.path.join(PROJECT_ROOT, "ext", "FoundationPose")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs")

CAM_NAME      = "cam_port"
PORT_GEOM     = "charge_port_visual"
PORT_BODY     = "charge_port_frame"
MESH_SCALE    = 1.0            # hummer_charge_port.obj: 이미 m 단위로 export (scene.xml 의 mesh scale 과 일치)
IMG_W, IMG_H  = 640, 480
EST_REFINE_ITER = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# MuJoCo 렌더링
# ---------------------------------------------------------------------------
def intrinsics_from_fovy(model: mujoco.MjModel, cam_id: int, w: int, h: int) -> np.ndarray:
    """수직 FOV(fovy) 기반 핀홀 카메라 intrinsic K (OpenCV 규약)."""
    fovy = np.deg2rad(model.cam_fovy[cam_id])
    fy = (h / 2.0) / np.tan(fovy / 2.0)
    fx = fy                       # MuJoCo 픽셀은 정사각 → fx == fy
    cx, cy = w / 2.0, h / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


def render_rgbd_mask(model, data, cam_id, geom_id):
    """RGB(uint8), depth(float32, meter), mask(bool) 를 cam_id 시점에서 추출."""
    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)

    # --- RGB ---
    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render().copy()                         # (H,W,3) uint8

    # --- Depth (meter) ---
    # mujoco.Renderer.enable_depth_rendering() 은 이미 선형 metric depth(미터)를
    # 반환하므로, CLAUDE.md 의 near/far 수동 변환식을 다시 적용하면 안 됩니다.
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam_id)
    depth = renderer.render().copy().astype(np.float32)    # (H,W) meter
    renderer.disable_depth_rendering()

    # 무한대(배경) 정리: FoundationPose 는 depth==0 을 invalid 로 취급
    depth[~np.isfinite(depth)] = 0.0
    depth[depth > 1e3] = 0.0

    # --- Segmentation mask (충전구 geom 만) ---
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=cam_id)
    seg = renderer.render().copy()                         # (H,W,2): [...,0]=objid
    renderer.disable_segmentation_rendering()
    mask = (seg[:, :, 0] == geom_id)

    return rgb, depth, mask


def gt_object_in_cam(model, data, cam_id, body_id) -> np.ndarray:
    """참고용 GT pose: 충전구 body 의 camera 좌표계(OpenCV) 4x4 변환."""
    # world ← body
    T_w_b = np.eye(4)
    T_w_b[:3, :3] = data.xmat[body_id].reshape(3, 3)
    T_w_b[:3, 3] = data.xpos[body_id]

    # world ← camera (MuJoCo 카메라: x=right, y=up, z=뒤쪽/-z 방향을 바라봄)
    T_w_c = np.eye(4)
    T_w_c[:3, :3] = data.cam_xmat[cam_id].reshape(3, 3)
    T_w_c[:3, 3] = data.cam_xpos[cam_id]

    # MuJoCo camera → OpenCV camera (y, z 부호 반전)
    mj_to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    T_w_c_cv = T_w_c @ mj_to_cv

    return np.linalg.inv(T_w_c_cv) @ T_w_b


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------
def save_inputs_figure(rgb, depth, mask, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(rgb);                ax[0].set_title("RGB (cam_port)");      ax[0].axis("off")
    d_vis = np.where(depth > 0, depth, np.nan)
    im = ax[1].imshow(d_vis, cmap="viridis"); ax[1].set_title("GT depth (m)"); ax[1].axis("off")
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    ax[2].imshow(mask, cmap="gray");  ax[2].set_title("Port mask");           ax[2].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print(f"[INFO] scene : {XML_PATH}")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PORT_GEOM)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PORT_BODY)
    if min(cam_id, geom_id, body_id) < 0:
        raise RuntimeError("scene.xml 에서 cam_port / charge_port_visual / "
                           "charge_port_frame 를 찾지 못했습니다.")

    rgb, depth, mask = render_rgbd_mask(model, data, cam_id, geom_id)
    K = intrinsics_from_fovy(model, cam_id, IMG_W, IMG_H)

    valid = depth[mask]
    valid = valid[valid > 0]
    print(f"[INFO] rgb {rgb.shape}  depth {depth.shape}  mask px={int(mask.sum())}")
    print(f"[INFO] K=\n{K}")
    if valid.size:
        print(f"[INFO] port depth  min={valid.min():.3f}  max={valid.max():.3f}  "
              f"mean={valid.mean():.3f} m")
    else:
        print("[WARN] 충전구 마스크 영역에 유효한 depth 가 없습니다 — 카메라/포트 배치 확인 필요")

    np.set_printoptions(precision=4, suppress=True)
    print(f"[INFO] (참고) GT object_in_cam =\n{gt_object_in_cam(model, data, cam_id, body_id)}")

    fig_path = os.path.join(OUTPUT_DIR, "foundation_pose_inputs.png")
    save_inputs_figure(rgb, depth, mask, fig_path)
    print(f"[INFO] 입력 시각화 저장: {fig_path}")

    # --- FoundationPose (CUDA 필요) ---
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    if not cuda_ok:
        print("\n[SKIP] CUDA 가 없어 FoundationPose 추정을 건너뜁니다. "
              "GPU 머신에서 동일 스크립트를 실행하면 6DoF pose 가 출력됩니다.")
        return

    pose = run_foundation_pose(rgb, depth, mask, K)
    print("\n[RESULT] FoundationPose ob_in_cam (4x4) =")
    print(pose)
    t = pose[:3, 3]
    print(f"[RESULT] translation (m): x={t[0]:.4f}  y={t[1]:.4f}  z={t[2]:.4f}")

    save_pose_overlay(rgb, depth, mask, K, pose)


def run_foundation_pose(rgb, depth, mask, K):
    """FoundationPose.register 로 ob_in_cam 6DoF 추정."""
    import trimesh
    sys.path.insert(0, FP_DIR)
    os.chdir(FP_DIR)            # FoundationPose 내부 상대 경로/import 대응

    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    from Utils import set_logging_format, set_seed
    import nvdiffrast.torch as dr

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(MESH_PATH)
    mesh.apply_scale(MESH_SCALE)          # mm → m (scene.xml 과 동일 스케일)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
        debug=1,
        debug_dir=os.path.join(OUTPUT_DIR, "fp_debug"),
    )

    pose = est.register(
        K=K.astype(np.float64),
        rgb=rgb,
        depth=depth.astype(np.float32),
        ob_mask=mask.astype(bool),
        iteration=EST_REFINE_ITER,
    )
    return pose


def save_pose_overlay(rgb, depth, mask, K, pose):
    """추정 pose 를 mesh bbox + 좌표축으로 RGB 위에 오버레이."""
    try:
        import trimesh, cv2
        from Utils import draw_posed_3d_box, draw_xyz_axis

        mesh = trimesh.load(MESH_PATH); mesh.apply_scale(MESH_SCALE)
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        center_pose = pose @ np.linalg.inv(to_origin)

        vis = draw_posed_3d_box(K, img=rgb.copy(), ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K,
                            thickness=3, transparency=0, is_input_rgb=True)
        out = os.path.join(OUTPUT_DIR, "foundation_pose_result.png")
        cv2.imwrite(out, vis[:, :, ::-1])
        print(f"[INFO] pose 오버레이 저장: {out}")
    except Exception as e:
        print(f"[WARN] pose 오버레이 시각화 실패: {e}")


if __name__ == "__main__":
    main()
