"""cam_port 단독 클로즈업 렌더 (AC 소켓 상단 여부 확인용). site geom은 숨김."""
import os, numpy as np, mujoco
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "models", "scene.xml"))
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, height=480, width=480)
opt = mujoco.MjvOption()
mujoco.mjv_defaultOption(opt)
opt.sitegroup[:] = 0  # site 숨김
r.update_scene(d, camera="cam_port", scene_option=opt)
rgb = r.render().copy()
plt.figure(figsize=(7, 7)); plt.imshow(rgb); plt.axis("off")
plt.title("cam_port (sites hidden) - AC socket should be UP")
out = os.path.join(ROOT, "outputs", "port_closeup.png")
plt.savefig(out, dpi=110, bbox_inches="tight"); print("saved:", out)
