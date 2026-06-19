#!/usr/bin/env python3
"""[Noise-2] MuJoCo segmentation renderer 로 충전구 seg_mask 추출 검증 + 노이즈 주입 데모.

- cam_port 에서 RGB / GT depth / segmentation 을 렌더한다.
- 충전구 geom(charge_port_visual) id 에 해당하는 픽셀 mask 를 추출한다.
- src/depth_noise.py 의 medium preset 으로 noisy depth 를 만들어 비교한다.
- 4분할(RGB+mask / seg_mask / GT depth / noisy depth) 을 outputs/seg_mask_test.png 로 저장.

Usage:
    conda activate wemeet
    python scripts/test_seg_mask.py
출력: outputs/seg_mask_test.png
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

from src.depth_noise import apply_preset, NOISE_PRESETS

SCENE = os.path.join(PROJECT_ROOT, "models", "scene.xml")
OUT = os.path.join(PROJECT_ROOT, "outputs", "seg_mask_test.png")

CAM = "cam_port"
PORT_GEOM = "charge_port_visual"     # 충전구 geom 이름 (scene.xml 에서 확인)
PRESET = "medium"
W, H = 640, 480
np.random.seed(0)


def main():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cam_id = model.camera(CAM).id
    geom_id = model.geom(PORT_GEOM).id
    print(f"[INFO] cam={CAM}  port geom='{PORT_GEOM}' id={geom_id}")

    r = mujoco.Renderer(model, height=H, width=W)

    # RGB
    r.update_scene(data, camera=CAM)
    rgb = r.render().copy()

    # GT depth (meter)
    r.enable_depth_rendering()
    r.update_scene(data, camera=CAM)
    depth = r.render().copy().astype(np.float32)
    r.disable_depth_rendering()
    depth[~np.isfinite(depth)] = 0.0
    depth[depth > 1e3] = 0.0

    # Segmentation → 충전구 geom mask
    r.enable_segmentation_rendering()
    r.update_scene(data, camera=CAM)
    seg = r.render().copy()          # (H, W, 2) int32: [...,0]=object id, [...,1]=object type
    r.disable_segmentation_rendering()
    mask = (seg[:, :, 0] == geom_id)

    # 검증
    px = int(mask.sum())
    print(f"[seg] mask px = {px}")
    if px == 0:
        print("[seg] FAIL — 충전구 mask 가 비어있음 (geom id / 카메라 가시성 확인)")
    else:
        ys, xs = np.where(mask)
        print(f"[seg] bbox x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]  "
              f"depth(port)= {depth[mask][depth[mask] > 0].mean():.3f} m")
        print(f"[seg] PASS — 충전구 mask 정상 추출")

    # 노이즈 주입 (medium): depth 에 gaussian+hole+quantize
    noisy_depth = apply_preset(depth, mask, preset=PRESET)
    valid = depth > 0
    holes_in_port = int((mask & (noisy_depth == 0)).sum())
    # 의미있는 비교: 포트(전경 ~1m) 영역 Δ. (경계 edge ring 이 far 배경에 닿으면 Δ 가 커지므로 분리)
    port_sel = mask & (noisy_depth > 0)
    port_diff = np.abs(noisy_depth - depth)[port_sel]
    print(f"[noise:{PRESET}] cfg={NOISE_PRESETS[PRESET]}")
    print(f"[noise:{PRESET}] 포트영역 hole px={holes_in_port} "
          f"({100*holes_in_port/max(px,1):.1f}%)  "
          f"포트 |Δdepth| mean={port_diff.mean()*1e3:.2f} mm "
          f"max={port_diff.max()*1e3:.2f} mm")

    # 시각화
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Port seg_mask extraction + depth noise (preset={PRESET})", fontsize=14)

    # RGB + mask overlay
    overlay = rgb.copy()
    overlay[mask] = (0.4 * overlay[mask] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
    ax[0, 0].imshow(overlay)
    ax[0, 0].set_title(f"{CAM} RGB + port mask (green)")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(mask, cmap="gray")
    ax[0, 1].set_title(f"seg_mask (geom={PORT_GEOM}, {px}px)")
    ax[0, 1].axis("off")

    # D435 동작범위(0.1~3.0 m)로 클립해 표시 — 원거리 배경은 제외, 포트/노이즈가 잘 보이도록
    VMIN, VMAX = 0.1, 3.0
    in_range = valid & (depth <= VMAX)
    dv = np.where(in_range, depth, np.nan)
    im1 = ax[1, 0].imshow(dv, cmap="turbo", vmin=VMIN, vmax=VMAX)
    ax[1, 0].set_title("GT depth (m, 0.1~3.0)")
    ax[1, 0].axis("off")
    fig.colorbar(im1, ax=ax[1, 0], fraction=0.046, pad=0.04)

    nd_range = (noisy_depth > 0) & (noisy_depth <= VMAX)
    nd = np.where(nd_range, noisy_depth, np.nan)
    im2 = ax[1, 1].imshow(nd, cmap="turbo", vmin=VMIN, vmax=VMAX)
    ax[1, 1].set_title(f"noisy depth ({PRESET}, 0.1~3.0) - holes=white")
    ax[1, 1].axis("off")
    fig.colorbar(im2, ax=ax[1, 1], fraction=0.046, pad=0.04)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=120)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
