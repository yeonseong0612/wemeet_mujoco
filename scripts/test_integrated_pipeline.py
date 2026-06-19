#!/usr/bin/env python3
"""FoundationPose ↔ IK 통합 테스트 + GT vs FP 정확도 평가 (Phase 1).

파이프라인 (CLAUDE.md §3):
    MuJoCo scene (충전구 랜덤 재배치)
        → cam_port RGB-D + GT mask 렌더
        → FoundationPose.register() → ob_in_cam (추정)
        → target_in_cam  = ob_in_cam   @ target_in_obj   (site 고정 오프셋)
        → target_in_world = cam_in_world @ target_in_cam
        → solve_pose_ik → UR10e 관절각

평가 (GT vs FP):
    ① ob_in_cam 자세오차      : translation(mm), rotation(deg)
    ② world 타겟 위치오차     : FP타겟 vs GT타겟 (mm)
    ③ 착지오차(landing)       : FP타겟으로 IK 푼 EE vs 실제 GT 타겟 (mm, deg)

충전구는 scene.xml 수정 없이 model.body_pos/body_quat 를 in-memory 로 바꿔
로봇 앞 도달가능 영역 안에서만 랜덤 배치한다(가시성·도달성 검증 통과분만 채택).

Usage:
    conda activate wemeet
    python scripts/test_integrated_pipeline.py            # 10 pose sweep + GUI 1건
    python scripts/test_integrated_pipeline.py --no-gui   # sweep 만
    python scripts/test_integrated_pipeline.py --n 20 --seed 1
"""
import os
import sys
import json
import argparse

import numpy as np
import mujoco

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

XML_PATH   = os.path.join(PROJECT_ROOT, "models", "scene.xml")
MESH_PATH  = os.path.join(PROJECT_ROOT, "models", "assets", "Port.obj")
FP_DIR     = os.path.join(PROJECT_ROOT, "ext", "FoundationPose")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
JSON_OUT   = os.path.join(OUTPUT_DIR, "integrated_eval.json")
PNG_OUT    = os.path.join(OUTPUT_DIR, "integrated_eval.png")

CAM_NAME   = "cam_port"
PORT_GEOM  = "charge_port_visual"
PORT_BODY  = "charge_port_frame"
EE_SITE    = "charger_tip_site"      # IK 대상 = 충전건 tip (custom_ee 프레임 +Y 0.204m 돌출)
TARGET_SITES = ["charge_port_approach", "charge_port_insert"]
OBS_SITE   = "charge_port_approach"  # eye-in-hand 관찰 pose: tip 을 여기 두고 카메라로 포트 관측

# object(port) 프레임 기준 charger_tip_site 목표 자세.
#   tip +X(총신 forward) → port -Z(소켓 안으로 삽입),  tip +Z(총구 up) → port +Y(위)
# 소켓 결합면은 object +Z(로봇쪽), 삽입은 -Z 방향. charger_tip_site 는 총신축 정렬 xyaxes 보유.
R_TIP_IN_OBJ = np.array([[0.0, -1.0, 0.0],
                          [0.0,  0.0, 1.0],
                          [-1.0, 0.0, 0.0]])
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

MESH_SCALE = 0.001
IMG_W, IMG_H = 640, 480                 # 기본(cam_port) 렌더 해상도
# 카메라별 렌더 해상도. cam_eih 는 Intel RealSense D435 네이티브(848x480).
CAM_RES = {"cam_port": (640, 480), "cam_eih": (848, 480)}
EST_REFINE_ITER = 5

# 랜덤화: 로봇 앞 도달가능 영역 (검증된 nominal (0,0.8,0.6) 주변)
POS_LO = np.array([-0.20, 0.70, 0.50])
POS_HI = np.array([ 0.20, 0.95, 0.70])
MAX_TILT_DEG = 10.0          # 실차 기본자세에서 ±각도 섭동
MIN_MASK_PX = 800            # 포트 가시성 하한
REACH_TOL = 2e-3             # insert 도달성 IK 허용오차 (m)
MAX_SAMPLE_TRIES = 400

os.makedirs(OUTPUT_DIR, exist_ok=True)

from src.ik import get_joint_ids, solve_position_ik, solve_pose_ik
from src.depth_noise import (NOISE_PRESETS, apply_depth_holes,
                             gaussian_depth_noise, quantize_depth)


def apply_noise_pipeline(depth, seg_mask, preset):
    """depth 에 D435 노이즈 프리셋(none/low/medium/high)을 주입 (CLAUDE.md §5).

    연결 순서: apply_depth_holes → gaussian_depth_noise → quantize_depth.
    'none' 또는 비활성 항목은 건너뛴다. 입력은 변형하지 않고 새 배열을 반환.
    seg_mask = 충전구 geom 픽셀 mask (hole 주입 영역).
    """
    cfg = NOISE_PRESETS[preset]
    out = depth.astype(np.float32, copy=True)
    if cfg["hole_rate"] > 0:
        out = apply_depth_holes(out, seg_mask, hole_rate=cfg["hole_rate"])
    if cfg["depth_gaussian_scale"] > 0:
        out = gaussian_depth_noise(out, scale=cfg["depth_gaussian_scale"])
    if cfg["quantize"]:
        out = quantize_depth(out)
    return out


# ---------------------------------------------------------------------------
# 기하/좌표 헬퍼
# ---------------------------------------------------------------------------
def intrinsics_from_fovy(model, cam_id, w, h):
    fovy = np.deg2rad(model.cam_fovy[cam_id])
    fy = (h / 2.0) / np.tan(fovy / 2.0)
    cx, cy = w / 2.0, h / 2.0
    return np.array([[fy, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def pos_mat_to_T(pos, mat):
    """(pos(3,), mat(3,3)) → 4x4 동차변환 (CLAUDE.md §2 규약)."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(mat).reshape(3, 3)
    T[:3, 3] = np.asarray(pos)
    return T


def cam_in_world_cv(model, data, cam_id):
    """카메라의 world pose (OpenCV 규약, 4x4) — 고정값이 아니라 호출 시점에 동적 계산.

    CLAUDE.md §2 의 eye-in-hand 규약대로, 캐시/상수 대신 매 호출마다
    data.cam_xpos / data.cam_xmat 을 읽어 카메라 pose 를 구한다. 따라서 카메라가
    EE 에 장착돼 로봇과 함께 움직이는 cam_eih 에서도, 외부 고정 cam_port 에서도
    동일 코드로 올바른 값을 돌려준다(cam_port 는 결과적으로 매번 같은 값).

        cam_pos = data.cam_xpos[cam_id]            # world frame
        cam_mat = data.cam_xmat[cam_id]            # world frame
        cam_in_world = pos_mat_to_T(cam_pos, cam_mat)

    ⚠️ cam_xpos/xmat 은 직전 mj_forward 의 qpos 로 산출된다. eye-in-hand 의 경우
       반드시 "영상을 캡처한 그 로봇 자세"에서(즉 IK 로 팔을 움직이기 전에) 호출해
       렌더 이미지와 카메라 pose 가 같은 상태를 가리키도록 해야 한다.
       여기서 IK 타겟은 world 프레임이므로 반환값(cam_in_world)이 곧 변환식의
       cam_in_base 역할을 한다(robot base 원점=world 원점).

    MuJoCo 카메라(-Z 시선,+Y up) → OpenCV(+Z 시선,-Y down) 변환은 diag(1,-1,-1).
    """
    cam_in_world = pos_mat_to_T(data.cam_xpos[cam_id], data.cam_xmat[cam_id])
    return cam_in_world @ np.diag([1.0, -1.0, -1.0, 1.0])


def gt_object_in_cam(model, data, cam_id, body_id):
    """GT object(body)_in_cam (OpenCV 4x4)."""
    T_w_b = np.eye(4)
    T_w_b[:3, :3] = data.xmat[body_id].reshape(3, 3)
    T_w_b[:3, 3] = data.xpos[body_id]
    return np.linalg.inv(cam_in_world_cv(model, data, cam_id)) @ T_w_b


def target_in_obj(model, sid):
    """site 의 object(body) 프레임 고정 오프셋 (4x4).
    위치=site 로컬 pos, 회전=R_TIP_IN_OBJ (charger_tip_site 삽입 자세)."""
    T = np.eye(4)
    T[:3, :3] = R_TIP_IN_OBJ
    T[:3, 3] = model.site_pos[sid]
    return T


def rot_err_deg(Ra, Rb):
    R = Ra.T @ Rb
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def rotvec_to_R(rv):
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def mat_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q                          # (w, x, y, z)


# 실차 장착 nominal: 소켓면 object +Z→world -Y(로봇쪽), AC위 object +Y→world +Z
# = 90° about world X (scene.xml charge_port_frame quat 과 동일)
R_NOMINAL = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


# ---------------------------------------------------------------------------
# 렌더 / 포즈 설정
# ---------------------------------------------------------------------------
def render_rgbd_mask(renderer, model, data, cam_id, geom_id):
    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam_id)
    depth = renderer.render().copy().astype(np.float32)
    renderer.disable_depth_rendering()
    depth[~np.isfinite(depth)] = 0.0
    depth[depth > 1e3] = 0.0

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=cam_id)
    seg = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    mask = (seg[:, :, 0] == geom_id)
    return rgb, depth, mask


def set_port_pose(model, data, body_id, pos, quat):
    model.body_pos[body_id] = pos
    model.body_quat[body_id] = quat
    mujoco.mj_forward(model, data)


def reset_robot(model, data):
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def set_observation_pose(model, data, ids):
    """charger_tip_site 를 OBS_SITE(charge_port_approach)에 삽입축 자세로 정렬 (2단계 IK).

    eye-in-hand 카메라(cam_eih)가 충전구를 정면으로 관측하도록 팔을 관찰 pose 에 둔다.
    (scripts/test_eih_view.py 의 set_observation_pose 와 동일 로직.)
    """
    R_world_obj = data.xmat[ids["body"]].reshape(3, 3)
    target_pos = data.site_xpos[ids["obs"]].copy()
    target_rot = R_world_obj @ R_TIP_IN_OBJ

    reset_robot(model, data)
    solve_position_ik(model, data, ids["ee"], target_pos,
                      ids["joints"], max_iters=400, verbose=False)
    q, _, _, pe, re = solve_pose_ik(
        model, data, ids["ee"], target_pos, target_rot,
        ids["joints"], max_iters=500, step_size=0.3, damping=1e-4,
        rot_weight=1.5, verbose=False)
    qpos_ids = [model.jnt_qposadr[j] for j in ids["joints"]]
    data.qpos[qpos_ids] = q
    mujoco.mj_forward(model, data)
    return pe, re


def position_for_capture(model, data, ids):
    """영상 캡처 직전 로봇 자세 설정.

    - eye-in-hand(cam_eih): 관찰 pose 로 이동(카메라가 포트를 보도록).
    - 외부 고정(cam_port): home 유지(팔이 포트를 가리지 않도록).
    """
    if ids["eye_in_hand"]:
        return set_observation_pose(model, data, ids)
    reset_robot(model, data)
    return None, None


def sample_reachable_pose(model, data, renderer, ids, rng):
    """로봇 앞 도달가능 + 포트 가시성 조건을 만족하는 (pos, quat) 샘플."""
    for _ in range(MAX_SAMPLE_TRIES):
        pos = rng.uniform(POS_LO, POS_HI)
        axis = rng.normal(size=3)
        axis /= (np.linalg.norm(axis) + 1e-12)
        ang = rng.uniform(0, np.deg2rad(MAX_TILT_DEG))
        R = rotvec_to_R(axis * ang) @ R_NOMINAL
        quat = mat_to_quat(R)

        set_port_pose(model, data, ids["body"], pos, quat)

        # 가시성: 실제 캡처와 동일한 로봇 자세(eye-in-hand=관찰 pose / 고정=home)에서 판정
        position_for_capture(model, data, ids)
        _, _, mask = render_rgbd_mask(renderer, model, data, ids["cam"], ids["geom"])
        if int(mask.sum()) < MIN_MASK_PX:
            continue

        # 도달성: insert site world 위치로 position-IK
        insert_pos = data.site_xpos[ids["insert"]].copy()
        reset_robot(model, data)
        _, _, pe = solve_position_ik(model, data, ids["ee"], insert_pos,
                                     ids["joints"], max_iters=400, verbose=False)
        reset_robot(model, data)
        if pe is not None and pe < REACH_TOL:
            set_port_pose(model, data, ids["body"], pos, quat)
            return pos.copy(), quat.copy()
    raise RuntimeError("도달가능/가시 포즈 샘플 실패 — 범위(POS_LO/HI) 확인 필요")


# ---------------------------------------------------------------------------
# FoundationPose
# ---------------------------------------------------------------------------
def build_estimator():
    import trimesh
    sys.path.insert(0, FP_DIR)
    os.chdir(FP_DIR)
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    from Utils import set_logging_format, set_seed
    import nvdiffrast.torch as dr
    import logging
    set_logging_format()
    logging.getLogger().setLevel(logging.WARNING)
    set_seed(0)

    mesh = trimesh.load(MESH_PATH)
    mesh.apply_scale(MESH_SCALE)
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
        scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
        glctx=dr.RasterizeCudaContext(), debug=0,
        debug_dir=os.path.join(OUTPUT_DIR, "fp_debug"))
    return est


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
def evaluate_pose(model, data, renderer, ids, est, K, site_offsets):
    """현재 포트 포즈에서 GT vs FP 비교 + IK 착지오차 측정. dict 반환."""
    # 렌더 직전 로봇을 캡처 자세로 배치(eye-in-hand=관찰 pose / 고정=home).
    # 영상 캡처와 카메라 pose 획득을 같은 로봇 자세에서 수행한다(eye-in-hand 정합 필수).
    # Twc_cv 는 이 시점의 data.cam_xpos/xmat 로 동적 계산되므로, 이후 IK 로 팔이
    # 움직여도(아래 루프) 캡처 당시 카메라 pose 스냅샷으로 유지된다.
    position_for_capture(model, data, ids)
    rgb, depth, mask = render_rgbd_mask(renderer, model, data, ids["cam"], ids["geom"])
    Twc_cv = cam_in_world_cv(model, data, ids["cam"])

    gt_ob = gt_object_in_cam(model, data, ids["cam"], ids["body"])

    # GT depth → seg_mask 기반 노이즈 주입 → noisy depth 를 FP 입력으로 사용 (GT pose 평가는 무관).
    depth_in = apply_noise_pipeline(depth, mask, ids["noise"])
    fp_ob = est.register(K=K.astype(np.float64), rgb=rgb,
                         depth=depth_in.astype(np.float32), ob_mask=mask.astype(bool),
                         iteration=EST_REFINE_ITER)

    rec = {
        "port_pos": data.xpos[ids["body"]].tolist(),
        "mask_px": int(mask.sum()),
        "obj_trans_err_mm": float(np.linalg.norm(fp_ob[:3, 3] - gt_ob[:3, 3]) * 1e3),
        "obj_rot_err_deg": rot_err_deg(gt_ob[:3, :3], fp_ob[:3, :3]),
        "targets": {},
    }

    for s in TARGET_SITES:
        T_obj = site_offsets[s]
        # GT 타겟(건 tip이 도달해야 할 실제 pose) vs FP 추정 타겟
        gt_world = Twc_cv @ gt_ob @ T_obj
        fp_world = Twc_cv @ fp_ob @ T_obj

        tgt_pos_err_mm = float(np.linalg.norm(fp_world[:3, 3] - gt_world[:3, 3]) * 1e3)

        # FP 타겟으로 IK → 최종 EE 가 실제(GT) 타겟에서 벗어난 착지오차
        reset_robot(model, data)
        solve_position_ik(model, data, ids["ee"], fp_world[:3, 3],
                          ids["joints"], max_iters=400, verbose=False)
        _, fpos, frot, pe, re = solve_pose_ik(
            model, data, ids["ee"], fp_world[:3, 3], fp_world[:3, :3],
            ids["joints"], max_iters=500, pos_tol=1e-4, rot_tol=1e-3,
            step_size=0.3, damping=1e-4, rot_weight=1.5, verbose=False)
        landing_pos_err_mm = float(np.linalg.norm(fpos - gt_world[:3, 3]) * 1e3)
        landing_rot_err_deg = rot_err_deg(frot, gt_world[:3, :3])
        reset_robot(model, data)

        rec["targets"][s] = {
            "target_pos_err_mm": tgt_pos_err_mm,
            "landing_pos_err_mm": landing_pos_err_mm,
            "landing_rot_err_deg": landing_rot_err_deg,
            "ik_pos_resid_mm": float(pe * 1e3) if pe is not None else None,
            "ik_converged": bool(pe is not None and pe < REACH_TOL),
        }
    return rec


def aggregate(records):
    def stats(vals):
        a = np.array([v for v in vals if v is not None], dtype=float)
        if a.size == 0:
            return {}
        return {"mean": float(a.mean()), "max": float(a.max()),
                "min": float(a.min()), "std": float(a.std())}
    agg = {
        "obj_trans_err_mm": stats([r["obj_trans_err_mm"] for r in records]),
        "obj_rot_err_deg": stats([r["obj_rot_err_deg"] for r in records]),
    }
    for s in TARGET_SITES:
        agg[f"{s}_target_pos_err_mm"] = stats([r["targets"][s]["target_pos_err_mm"] for r in records])
        agg[f"{s}_landing_pos_err_mm"] = stats([r["targets"][s]["landing_pos_err_mm"] for r in records])
        agg[f"{s}_landing_rot_err_deg"] = stats([r["targets"][s]["landing_rot_err_deg"] for r in records])
    return agg


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------
def save_png(records, agg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(records)
    idx = np.arange(n)
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"FoundationPose vs GT — integrated pipeline eval (N={n})", fontsize=14)

    ax[0, 0].bar(idx, [r["obj_trans_err_mm"] for r in records], color="tab:blue")
    ax[0, 0].set_title("ob_in_cam translation error (mm)")
    ax[0, 0].set_xlabel("pose #"); ax[0, 0].axhline(
        agg["obj_trans_err_mm"]["mean"], color="k", ls="--", lw=1,
        label=f"mean {agg['obj_trans_err_mm']['mean']:.2f}")
    ax[0, 0].legend()

    ax[0, 1].bar(idx, [r["obj_rot_err_deg"] for r in records], color="tab:orange")
    ax[0, 1].set_title("ob_in_cam rotation error (deg)")
    ax[0, 1].set_xlabel("pose #"); ax[0, 1].axhline(
        agg["obj_rot_err_deg"]["mean"], color="k", ls="--", lw=1,
        label=f"mean {agg['obj_rot_err_deg']['mean']:.3f}")
    ax[0, 1].legend()

    w = 0.38
    for j, s in enumerate(TARGET_SITES):
        vals = [r["targets"][s]["landing_pos_err_mm"] for r in records]
        ax[1, 0].bar(idx + (j - 0.5) * w, vals, width=w, label=s.replace("charge_port_", ""))
    ax[1, 0].set_title("landing position error: FP-IK EE vs true target (mm)")
    ax[1, 0].set_xlabel("pose #"); ax[1, 0].legend()

    ax[1, 1].axis("off")
    lines = ["Aggregate (mean / max / std)", ""]
    label_map = {
        "obj_trans_err_mm": "ob_in_cam trans (mm)",
        "obj_rot_err_deg":  "ob_in_cam rot  (deg)",
        "charge_port_insert_landing_pos_err_mm": "insert landing (mm)",
        "charge_port_insert_landing_rot_err_deg": "insert landing rot (deg)",
        "charge_port_approach_landing_pos_err_mm": "approach landing (mm)",
    }
    for k, lab in label_map.items():
        st = agg.get(k, {})
        if st:
            lines.append(f"{lab:24s}: {st['mean']:7.3f} / {st['max']:7.3f} / {st['std']:6.3f}")
    ax[1, 1].text(0.0, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=11)

    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# GUI (한 케이스 애니메이션)
# ---------------------------------------------------------------------------
def run_gui(model, data, ids, est, K, site_offsets, renderer, pose):
    import time
    import mujoco.viewer

    pos, quat = pose
    set_port_pose(model, data, ids["body"], pos, quat)
    # 렌더 직전 캡처 자세 배치 후, 그 자세에서 카메라 pose 를 동적 스냅샷 (eye-in-hand 정합).
    position_for_capture(model, data, ids)
    rgb, depth, mask = render_rgbd_mask(renderer, model, data, ids["cam"], ids["geom"])
    Twc_cv = cam_in_world_cv(model, data, ids["cam"])
    depth_in = apply_noise_pipeline(depth, mask, ids["noise"])
    fp_ob = est.register(K=K.astype(np.float64), rgb=rgb,
                         depth=depth_in.astype(np.float32), ob_mask=mask.astype(bool),
                         iteration=EST_REFINE_ITER)

    # FP 추정 → approach / insert world 타겟 → IK (2단계: 위치 우선 → 자세)
    q_targets = []
    for s in TARGET_SITES:
        fp_world = Twc_cv @ fp_ob @ site_offsets[s]
        reset_robot(model, data)
        solve_position_ik(model, data, ids["ee"], fp_world[:3, 3],
                          ids["joints"], max_iters=400, verbose=False)
        q, _, _, pe, re = solve_pose_ik(
            model, data, ids["ee"], fp_world[:3, 3], fp_world[:3, :3],
            ids["joints"], max_iters=500, step_size=0.3, damping=1e-4,
            rot_weight=1.5, verbose=False)
        q_targets.append(q)
        print(f"[GUI] {s}: IK pos_resid={pe*1e3:.2f} mm  rot_resid={re:.4f}")

    qpos_ids = [model.jnt_qposadr[j] for j in ids["joints"]]
    reset_robot(model, data)
    q_start = data.qpos[qpos_ids].copy()
    segments = [q_start] + q_targets
    seg_dur = [4.0, 4.0]                      # start→approach, approach→insert

    print("[GUI] 뷰어 시작 — FP 추정 타겟으로 approach→insert 이동")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        reset_robot(model, data)
        t0 = time.time()
        while viewer.is_running():
            el = time.time() - t0
            acc, q_cmd = 0.0, segments[-1]
            for i, dur in enumerate(seg_dur):
                if acc <= el < acc + dur:
                    a = (el - acc) / dur
                    q_cmd = (1 - a) * segments[i] + a * segments[i + 1]
                    break
                acc += dur
            data.ctrl[:len(q_cmd)] = q_cmd
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="평가 포즈 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gui", action="store_true", help="GUI 생략")
    ap.add_argument("--camera", default=CAM_NAME, choices=["cam_port", "cam_eih"],
                    help="입력 카메라 (기본 cam_port=외부고정 / cam_eih=eye-in-hand)")
    ap.add_argument("--noise", default="none", choices=list(NOISE_PRESETS),
                    help="depth 노이즈 프리셋 (CLAUDE.md §5: none/low/medium/high)")
    ap.add_argument("--out", default=None,
                    help="결과 JSON 출력 경로 (PNG는 같은 stem .png). 미지정 시 camera/noise 기반 자동 이름")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)            # depth_noise 의 np.random 재현성 확보
    cwd0 = os.getcwd()

    img_w, img_h = CAM_RES.get(args.camera, (IMG_W, IMG_H))
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    # offscreen framebuffer 를 렌더 해상도에 맞게 키운다 (scene.xml 수정 없이 in-memory).
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, img_w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, img_h)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=img_h, width=img_w)

    ids = {
        "cam": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera),
        "geom": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PORT_GEOM),
        "body": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PORT_BODY),
        "ee": model.site(EE_SITE).id,
        "joints": get_joint_ids(model, JOINT_NAMES),
        "targets": {s: model.site(s).id for s in TARGET_SITES},
    }
    ids["insert"] = ids["targets"]["charge_port_insert"]
    ids["obs"] = model.site(OBS_SITE).id
    ids["noise"] = args.noise
    # eye-in-hand 여부: 카메라가 worldbody(고정) 가 아니라 로봇 body 에 장착됐는지로 판정.
    ids["eye_in_hand"] = bool(model.cam_bodyid[ids["cam"]] != 0)
    site_offsets = {s: target_in_obj(model, ids["targets"][s]) for s in TARGET_SITES}
    K = intrinsics_from_fovy(model, ids["cam"], img_w, img_h)
    print(f"[INFO] camera={args.camera}  eye_in_hand={ids['eye_in_hand']}  "
          f"fovy={model.cam_fovy[ids['cam']]}  res={img_w}x{img_h}  noise={args.noise}")

    # 1) 도달가능 포즈 N개 사전 샘플 (FP 빌드 전, MuJoCo만 사용)
    print(f"[INFO] 도달가능 포즈 {args.n}개 샘플링 ...")
    poses = [sample_reachable_pose(model, data, renderer, ids, rng) for _ in range(args.n)]
    print(f"[INFO] 샘플 완료. FoundationPose 초기화 ...")

    # 2) FP 빌드 (1회) — 이후 cwd 는 FP_DIR
    est = build_estimator()

    # 3) sweep 평가
    records = []
    for i, (pos, quat) in enumerate(poses):
        set_port_pose(model, data, ids["body"], pos, quat)
        rec = evaluate_pose(model, data, renderer, ids, est, K, site_offsets)
        records.append(rec)
        t = rec["targets"]["charge_port_insert"]
        print(f"[{i+1}/{args.n}] ob_in_cam: {rec['obj_trans_err_mm']:6.2f} mm / "
              f"{rec['obj_rot_err_deg']:5.2f}°  | insert landing: "
              f"{t['landing_pos_err_mm']:6.2f} mm / {t['landing_rot_err_deg']:5.2f}°")

    agg = aggregate(records)
    result = {"config": {"n": args.n, "seed": args.seed,
                         "camera": args.camera, "eye_in_hand": ids["eye_in_hand"],
                         "noise": args.noise, "noise_cfg": NOISE_PRESETS[args.noise],
                         "img_w": img_w, "img_h": img_h,
                         "pos_lo": POS_LO.tolist(), "pos_hi": POS_HI.tolist(),
                         "max_tilt_deg": MAX_TILT_DEG, "refine_iter": EST_REFINE_ITER},
              "aggregate": agg, "per_pose": records}

    os.chdir(cwd0)
    if args.out:
        json_out = args.out
        png_out = os.path.splitext(args.out)[0] + ".png"
    else:
        # cam_port + noise=none 은 canonical 파일명 유지(베이스라인 보존), 그 외는 접미사로 분리.
        tag = []
        if args.camera != "cam_port":
            tag.append(args.camera)
        if args.noise != "none":
            tag.append(args.noise)
        suffix = ("_" + "_".join(tag)) if tag else ""
        json_out = JSON_OUT.replace(".json", f"{suffix}.json")
        png_out = PNG_OUT.replace(".png", f"{suffix}.png")
    os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(result, f, indent=2)
    save_png(records, agg, png_out)

    print("\n===== Aggregate (mean / max / std) =====")
    print(f"  ob_in_cam trans : {agg['obj_trans_err_mm']['mean']:.2f} / "
          f"{agg['obj_trans_err_mm']['max']:.2f} / {agg['obj_trans_err_mm']['std']:.2f} mm")
    print(f"  ob_in_cam rot   : {agg['obj_rot_err_deg']['mean']:.3f} / "
          f"{agg['obj_rot_err_deg']['max']:.3f} / {agg['obj_rot_err_deg']['std']:.3f} deg")
    print(f"  insert landing  : {agg['charge_port_insert_landing_pos_err_mm']['mean']:.2f} / "
          f"{agg['charge_port_insert_landing_pos_err_mm']['max']:.2f} mm")
    print(f"\n[INFO] 저장: {json_out}\n[INFO] 저장: {png_out}")

    # 4) GUI 한 케이스 (랜덤 도달가능 포즈)
    if not args.no_gui:
        os.chdir(FP_DIR)
        gui_pose = poses[int(rng.integers(len(poses)))]
        try:
            run_gui(model, data, ids, est, K, site_offsets, renderer, gui_pose)
        except Exception as e:
            print(f"[WARN] GUI 실행 실패(디스플레이 없음?): {e}")
        finally:
            os.chdir(cwd0)


if __name__ == "__main__":
    main()
