#!/usr/bin/env python3
"""[EiH-3/EiH-4] Eye-in-Hand 초기 관찰 pose 설정 + 3분할 뷰어.

목적:
    cam_eih(custom_ee 장착) FOV 안에 충전구 전체가 들어오는 초기 관찰 pose를
    설정하고, overview / cam_eih RGB / cam_eih depth 를 3분할로 시각화한다.

초기 관찰 pose (GT 기준):
    charger_tip_site 를 charge_port_approach (소켓 앞 0.15 m) 에 두고,
    삽입축 자세(R_TIP_IN_OBJ)로 정렬 → 2단계 IK(위치 우선 → 자세).
    이때 cam_eih 는 tip 위쪽(+Z)에서 전방(+Y)을 ~16° 하향으로 내려다본다.

검증:
    cam_eih 에서 충전구 geom seg mask 를 렌더 → bbox 가 화면 경계에 닿지 않고
    (전체 포함), 단축 px >= MIN_SHORT_PX 인지 확인한다.

Usage:
    conda activate wemeet
    python scripts/test_eih_view.py
출력: outputs/eih_view_test.png
"""
import os
import sys

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.ik import get_joint_ids, solve_position_ik, solve_pose_ik

SCENE = os.path.join(PROJECT_ROOT, "models", "scene.xml")
OUT = os.path.join(PROJECT_ROOT, "outputs", "eih_view_test.png")

EE_SITE = "charger_tip_site"
PORT_BODY = "charge_port_frame"
PORT_GEOM = "charge_port_visual"
OBS_SITE = "charge_port_approach"      # 초기 관찰 시 tip 을 둘 site (소켓 앞 0.15 m)
EIH_CAM = "cam_eih"
OVERVIEW_CAM = "overview_camera"
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

# object(port) 프레임 기준 charger_tip_site 목표 자세 (test_integrated_pipeline.py 와 동일).
#   tip +X(총신 forward) → port -Z(소켓 안으로 삽입),  tip +Z(총구 up) → port +Y(위)
R_TIP_IN_OBJ = np.array([[0.0, -1.0, 0.0],
                         [0.0,  0.0, 1.0],
                         [-1.0, 0.0, 0.0]])

# D435 기준 cam_eih 해상도 / overview 해상도
EIH_W, EIH_H = 848, 480
OV_W, OV_H = 640, 480
MIN_SHORT_PX = 80          # FoundationPose 입력 권장 충전구 단축 하한


def reset_robot(model, data):
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def set_observation_pose(model, data, ids):
    """charger_tip_site 를 charge_port_approach 에 삽입축 자세로 정렬 (2단계 IK)."""
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


def render_rgb(renderer, data, cam):
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()


def render_depth(renderer, data, cam):
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam)
    depth = renderer.render().copy().astype(np.float32)
    renderer.disable_depth_rendering()
    depth[~np.isfinite(depth)] = 0.0
    depth[depth > 1e3] = 0.0
    return depth


def render_port_mask(renderer, data, cam, geom_id):
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=cam)
    seg = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    return seg[:, :, 0] == geom_id


def check_full_in_view(mask):
    """포트 mask 의 화면 포함 여부 / bbox / 단축 px 점검."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {"in_view": False, "reason": "포트가 FOV 밖 (mask 0px)"}
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    H, W = mask.shape
    touches = (x0 == 0 or y0 == 0 or x1 == W - 1 or y1 == H - 1)
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    return {
        "in_view": (not touches),
        "bbox": (int(x0), int(y0), int(x1), int(y1)),
        "bbox_wh": (int(bw), int(bh)),
        "short_px": int(min(bw, bh)),
        "px": int(mask.sum()),
        "touches_border": bool(touches),
    }


def main():
    model = mujoco.MjModel.from_xml_path(SCENE)
    # offscreen framebuffer 를 D435 폭(848)에 맞게 키운다 (scene.xml 수정 없이 in-memory).
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, EIH_W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, EIH_H)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ids = {
        "ee": model.site(EE_SITE).id,
        "obs": model.site(OBS_SITE).id,
        "body": model.body(PORT_BODY).id,
        "geom": model.geom(PORT_GEOM).id,
        "joints": get_joint_ids(model, JOINT_NAMES),
    }

    port_pos = data.xpos[ids["body"]].copy()
    print(f"[port] body pose (nominal scene): pos={np.round(port_pos, 3)}")

    # 초기 관찰 pose 설정
    pe, re = set_observation_pose(model, data, ids)
    print(f"[obs-pose] IK 잔차: pos={pe * 1e3:.2f} mm  rot={re:.4f}")
    cam_id = model.camera(EIH_CAM).id
    print(f"[cam_eih] world pos={np.round(data.cam_xpos[cam_id], 3)}  "
          f"fovy={model.cam_fovy[cam_id]}  res={EIH_W}x{EIH_H}")

    # 렌더 (cam_eih: D435 해상도, overview: 별도 해상도 → 각각 renderer)
    r_eih = mujoco.Renderer(model, height=EIH_H, width=EIH_W)
    r_ov = mujoco.Renderer(model, height=OV_H, width=OV_W)

    overview = render_rgb(r_ov, data, OVERVIEW_CAM)
    rgb_eih = render_rgb(r_eih, data, EIH_CAM)
    depth_eih = render_depth(r_eih, data, EIH_CAM)
    mask_eih = render_port_mask(r_eih, data, EIH_CAM, ids["geom"])

    # 검증
    chk = check_full_in_view(mask_eih)
    print(f"[검증] {chk}")
    if chk["in_view"]:
        ok_short = chk["short_px"] >= MIN_SHORT_PX
        print(f"[검증] 충전구 전체 FOV 포함: PASS  "
              f"(단축 {chk['short_px']}px {'OK' if ok_short else f'< {MIN_SHORT_PX}px 권장미달'})")
    else:
        print(f"[검증] 충전구 전체 FOV 포함: FAIL — {chk.get('reason', '경계에 닿음')}")

    valid = depth_eih > 0
    if valid.any():
        near = valid & (depth_eih < 2.0)
        print(f"[depth] cam_eih min={depth_eih[valid].min():.3f} "
              f"max={depth_eih[valid].max():.3f} m  "
              f"port부근(<2m) mean={depth_eih[near].mean():.3f} m  px={int(near.sum())}")

    # 3분할 시각화
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    ax[0].imshow(overview)
    ax[0].set_title("Overview (obs pose)")
    ax[0].axis("off")

    ax[1].imshow(rgb_eih)
    ax[1].set_title(f"cam_eih RGB  {EIH_W}x{EIH_H}")
    # 포트 bbox 오버레이
    if "bbox" in chk:
        x0, y0, x1, y1 = chk["bbox"]
        color = "lime" if chk["in_view"] else "red"
        ax[1].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                      fill=False, edgecolor=color, linewidth=2))
        ax[1].text(x0, max(y0 - 6, 0), f"port {chk['bbox_wh'][0]}x{chk['bbox_wh'][1]}px",
                   color=color, fontsize=9, weight="bold")
    ax[1].axis("off")

    # D435 동작 범위(near 0.1 m ~ far 3.0 m) 밖은 배경으로 보고 표시 제외
    in_range = valid & (depth_eih >= 0.1) & (depth_eih <= 3.0)
    depth_vis = np.where(in_range, depth_eih, np.nan)
    im = ax[2].imshow(depth_vis, cmap="turbo", vmin=0.1, vmax=3.0)
    ax[2].set_title("cam_eih Depth (m, 0.1~3.0)")
    ax[2].axis("off")
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=120)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
