#!/usr/bin/env python3
"""
MuJoCo 스테레오 카메라 → FastACV 깊이 추정 검증 스크립트.
Usage: python scripts/test_stereo_depth.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import mujoco

from ext.fastacv import FastNet

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH    = os.path.join(PROJECT_ROOT, "models", "scene.xml")
CKPT_PATH   = os.path.join(PROJECT_ROOT, "checkpoints", "fastacvnetplus_kitti2015.ckpt")
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "outputs")

BASELINE  = 0.12   # meters — must match scene.xml stereo camera X offset * 2
FOVY_DEG  = 45.0   # must match scene.xml fovy
IMG_W     = 640
IMG_H     = 480

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_fastnet(ckpt_path: str, device: torch.device) -> FastNet:
    net = FastNet(cfg=None)
    net.to(device)

    raw = torch.load(ckpt_path, map_location=device)
    if isinstance(raw, dict):
        state_dict = raw.get('state_dict', raw.get('model', raw))
    else:
        state_dict = raw

    # strip DataParallel 'module.' prefix if present
    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys  ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
    net.eval()
    return net


def render_stereo(model: mujoco.MjModel, data: mujoco.MjData):
    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)

    left_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "stereo_left")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "stereo_right")

    if left_id == -1 or right_id == -1:
        raise RuntimeError("scene.xml 에 'stereo_left' / 'stereo_right' 카메라가 없습니다.")

    renderer.update_scene(data, camera=left_id)
    left_img = renderer.render().copy()   # (H, W, 3) uint8

    renderer.update_scene(data, camera=right_id)
    right_img = renderer.render().copy()

    return left_img, right_img


def to_tensor(img_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H,W,3) uint8  →  (1,3,H,W) float32 [0,1]"""
    t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def pad32(t: torch.Tensor):
    """Pad spatial dims to be divisible by 32; return padded tensor + original (h,w)."""
    _, _, h, w = t.shape
    ph = (32 - h % 32) % 32
    pw = (32 - w % 32) % 32
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph))
    return t, h, w


def disp_to_depth(disp: torch.Tensor, focal_px: float, baseline_m: float) -> torch.Tensor:
    depth = torch.zeros_like(disp)
    mask = disp > 0.5
    depth[mask] = focal_px * baseline_m / disp[mask]
    return depth


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] device: {device}")

    # --- MuJoCo 로드 및 스테레오 렌더링 ---
    print(f"[INFO] XML: {XML_PATH}")
    mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    mj_data  = mujoco.MjData(mj_model)
    mujoco.mj_step(mj_model, mj_data)

    left_img, right_img = render_stereo(mj_model, mj_data)
    print(f"[INFO] 렌더링 완료: {left_img.shape}")

    # --- FastNet 로드 ---
    print(f"[INFO] 체크포인트: {CKPT_PATH}")
    net = load_fastnet(CKPT_PATH, device)
    print("[INFO] FastNet 로드 완료")

    # --- 전처리 ---
    left_t,  = (to_tensor(left_img,  device),)
    right_t, = (to_tensor(right_img, device),)
    left_pad,  orig_h, orig_w = pad32(left_t)
    right_pad, _,      _      = pad32(right_t)

    # --- 추론 ---
    with torch.no_grad():
        out = net({'left_i': left_pad, 'right_i': right_pad}, eval=True)

    disp = out['disp'][:, :orig_h, :orig_w]   # (1, H, W) pixels

    # --- disparity → depth ---
    # focal_px: 핀홀 카메라, fovy 기준 수직 초점거리
    focal_px = (IMG_H / 2.0) / np.tan(np.radians(FOVY_DEG / 2.0))
    print(f"[INFO] focal_px = {focal_px:.1f}  baseline = {BASELINE} m")

    depth = disp_to_depth(disp, focal_px, BASELINE)

    disp_np  = disp.squeeze().cpu().numpy()
    depth_np = depth.squeeze().cpu().numpy()

    valid = depth_np[depth_np > 0]
    print(f"[INFO] Disparity  min={disp_np.min():.2f}  max={disp_np.max():.2f} px")
    if valid.size:
        print(f"[INFO] Depth      min={valid.min():.3f}  max={valid.max():.3f} m  "
              f"mean={valid.mean():.3f} m")
    else:
        print("[WARN] 유효한 depth 값이 없습니다 — 장면이 너무 단순하거나 카메라 설정을 확인하세요.")

    # --- 시각화 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("MuJoCo Stereo → FastACV Depth Estimation", fontsize=14)

    axes[0, 0].imshow(left_img);  axes[0, 0].set_title("Left Image");  axes[0, 0].axis('off')
    axes[0, 1].imshow(right_img); axes[0, 1].set_title("Right Image"); axes[0, 1].axis('off')

    im_d = axes[1, 0].imshow(disp_np, cmap='plasma')
    axes[1, 0].set_title("Disparity (px)"); axes[1, 0].axis('off')
    plt.colorbar(im_d, ax=axes[1, 0], fraction=0.046)

    depth_vis = np.clip(depth_np, 0, 5)
    im_z = axes[1, 1].imshow(depth_vis, cmap='viridis')
    axes[1, 1].set_title("Depth (m, clipped 5 m)"); axes[1, 1].axis('off')
    plt.colorbar(im_z, ax=axes[1, 1], fraction=0.046)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "stereo_depth_test.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] 결과 저장: {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
