#!/usr/bin/env python3
"""RGB-D 시각화 검증: overview + cam_port RGB + cam_port GT depth(meter).

Usage:
    conda activate pytorch
    python scripts/test_rgbd_view.py
출력: outputs/rgbd_view_test.png
"""
import os

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "models", "scene.xml")
OUT = os.path.join(ROOT, "outputs", "rgbd_view_test.png")
W, H = 640, 480

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

r = mujoco.Renderer(model, height=H, width=W)

# 1) 전체 조망
r.update_scene(data, camera="overview_camera")
overview = r.render().copy()

# 2) RGB-D 카메라(cam_port) — RGB
r.update_scene(data, camera="cam_port")
rgb = r.render().copy()

# 3) RGB-D 카메라(cam_port) — Depth (meter)
r.enable_depth_rendering()
r.update_scene(data, camera="cam_port")
depth = r.render().copy()          # float32, meters
r.disable_depth_rendering()

valid = np.isfinite(depth) & (depth > 0)
print(f"[depth] min={depth[valid].min():.3f} m  "
      f"max={depth[valid].max():.3f} m  "
      f"mean={depth[valid].mean():.3f} m")

# 충전구 표면(원경 배경 floor 제외) 통계: 2 m 이내만
near = valid & (depth < 2.0)
if near.any():
    print(f"[depth<2m] min={depth[near].min():.3f} m  "
          f"max={depth[near].max():.3f} m  "
          f"mean={depth[near].mean():.3f} m  px={int(near.sum())}")

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(overview); ax[0].set_title("Overview"); ax[0].axis("off")
ax[1].imshow(rgb);      ax[1].set_title("cam_port RGB"); ax[1].axis("off")
depth_vis = np.where(valid, depth, np.nan)
im = ax[2].imshow(depth_vis, cmap="turbo")
ax[2].set_title("cam_port Depth (m)"); ax[2].axis("off")
fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.tight_layout(); plt.savefig(OUT, dpi=120)
print(f"saved: {OUT}")
