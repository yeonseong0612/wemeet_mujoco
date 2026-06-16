"""EE.obj 충전건만 독립 모델로 렌더 — 커넥터(멀티핀) 면이 어느 끝인지 확인.
custom_ee 프레임 좌표축(빨강=+X, 초록=+Y, 파랑=+Z)과 현재 tip(흰 구) 표시."""
import os
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/yskim/projects/wemeet_mujoco"

XML = f"""
<mujoco>
  <compiler angle="radian" meshdir="{ROOT}/models/robots/ur10e/assets"/>
  <asset>
    <mesh name="ee" file="EE.obj" scale="0.001 0.001 0.001"/>
  </asset>
  <visual><global offwidth="640" offheight="520"/></visual>
  <worldbody>
    <light pos="0.3 0.3 0.5" dir="-1 -1 -1.5"/>
    <light pos="-0.3 -0.3 0.5" dir="1 1 -1.5"/>
    <!-- geom 변환은 ur10e.xml 와 동일: pos 0.08 0.1 -0.03, euler 1.5708 0 0 -->
    <body name="ee" pos="0 0 0">
      <geom type="mesh" mesh="ee" pos="0.08 0.1 -0.03" euler="1.5708 0 0" rgba="0.3 0.6 0.9 1"/>
      <!-- custom_ee 좌표축 -->
      <site name="ax" pos="0.06 0 0" size="0.06 0.003 0.003" type="box" rgba="1 0 0 1" euler="0 0 0"/>
      <site name="ay" pos="0 0.06 0" size="0.003 0.06 0.003" type="box" rgba="0 1 0 1"/>
      <site name="az" pos="0 0 0.06" size="0.003 0.003 0.06" type="box" rgba="0 0 1 1"/>
      <!-- 현재 charger_tip_site 위치(흰 구) / 반대편 끝(노랑) -->
      <site name="tip" pos="0.0005 -0.0256 -0.0823" size="0.012" rgba="1 1 1 1"/>
      <site name="other" pos="0 0.208 0.012" size="0.012" rgba="1 0.8 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

opt = mujoco.MjvOption()
opt.sitegroup[:] = 1

renderer = mujoco.Renderer(model, height=520, width=640)
fig, ax = plt.subplots(2, 2, figsize=(13, 11))
views = [(0, -5), (90, -5), (180, -5), (90, -80)]
for k, (az, el) in enumerate(views):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.06, -0.02]
    cam.distance, cam.azimuth, cam.elevation = 0.42, az, el
    renderer.update_scene(data, camera=cam, scene_option=opt)
    a = ax[k // 2, k % 2]
    a.imshow(renderer.render()); a.axis("off")
    a.set_title(f"az={az} el={el}  (R=+X G=+Y B=+Z, white=현재tip, yellow=반대끝)")
out = os.path.join(ROOT, "outputs/gun_alone.png")
plt.tight_layout(); plt.savefig(out, dpi=110)
print("saved:", out)
