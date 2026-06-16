"""홈 자세에서 충전건만 가까이 렌더 — 실제 커넥터(결합) 면이 어느 끝인지 확인.
charger_tip_site(녹) / charger_axis_site(청) 마커 위치도 함께 표시."""
import os, sys
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/yskim/projects/wemeet_mujoco"
model = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "models/scene.xml"))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# 사이트 group 4 가 보이도록 옵션 설정
opt = mujoco.MjvOption()
opt.sitegroup[4] = 1

tip = data.site_xpos[model.site("charger_tip_site").id]
print("charger_tip_site world:", np.round(tip, 4))

renderer = mujoco.Renderer(model, height=480, width=640)
fig, ax = plt.subplots(1, 3, figsize=(20, 6))
for k, az in enumerate([60, 150, 240]):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = tip
    cam.distance, cam.azimuth, cam.elevation = 0.35, az, -10
    renderer.update_scene(data, camera=cam, scene_option=opt)
    ax[k].imshow(renderer.render()); ax[k].axis("off")
    ax[k].set_title(f"az={az}")
out = os.path.join(ROOT, "outputs/gun_closeup.png")
plt.tight_layout(); plt.savefig(out, dpi=120)
print("saved:", out)
